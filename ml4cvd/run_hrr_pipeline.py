import os
import pandas as pd
import datetime
import numpy as np
from multiprocessing import cpu_count
import logging

from ml4cvd.recipes import _predict_scalars_and_evaluate_from_generator
from ml4cvd.logger import load_config
from ml4cvd.tensor_generators import test_train_valid_tensor_generators, big_batch_from_minibatch_generator
from ml4cvd.models import make_multimodal_multitask_model, BottleneckType, train_model_from_generators
from ml4cvd.exercise_ecg_tensormaps import OUTPUT_FOLDER, USER, FIGURE_FOLDER, BIOSPPY_MEASUREMENTS_PATH, RECOVERY_MODEL_PATH, TENSOR_FOLDER, RECOVERY_MODEL_ID
from ml4cvd.exercise_ecg_tensormaps import build_hr_biosppy_measurements_csv, plot_hr_from_biosppy_summary_stats
from ml4cvd.exercise_ecg_tensormaps import ecg_bike_recovery_downsampled8x, _make_hr_biosppy_tmaps

SEED = 217
REMAKE_LABELS = False or not os.path.exists(BIOSPPY_MEASUREMENTS_PATH)
RETRAIN_RECOVERY_MODEL = False or not os.path.exists(RECOVERY_MODEL_PATH)


def train_recovery_model():
    """trains model to get biosppy measurements from recovery"""
    patience = 16
    epochs = 100
    batch_size = 128
    valid_ratio = .1
    test_ratio = .1
    data = pd.read_csv(BIOSPPY_MEASUREMENTS_PATH)
    data_set_size = (len(data) - len(data['error'].dropna())) // batch_size  # approximation
    training_steps = int(data_set_size * (1 - valid_ratio - test_ratio))
    validation_steps = int(data_set_size * valid_ratio / 2)
    test_steps = int(data_set_size * test_ratio)

    hr_tmaps, hrr_tmaps = _make_hr_biosppy_tmaps()
    tmaps_in = [ecg_bike_recovery_downsampled8x]
    tmaps_out = list(hr_tmaps.values()) + list(hrr_tmaps.values())
    model = make_multimodal_multitask_model(
        tensor_maps_in=tmaps_in,
        tensor_maps_out=tmaps_out,
        activation='swish',
        learning_rate=1e-3,
        bottleneck_type=BottleneckType.FlattenRestructure,
        optimizer='radam',
        dense_layers=[64, 64],
        conv_layers=[32, 32, 32, 32, 32],  # lots of residual blocks with dilation
        dense_blocks=[32, 32, 32],
        block_size=3,
        conv_normalize='batch_norm',
        conv_x=16,
        conv_dilate=True,
        pool_x=4,
        pool_type='max',
        conv_type='conv',
    )
    workers = cpu_count() * 2
    generate_train, generate_valid, generate_test = test_train_valid_tensor_generators(
        tensor_maps_in=tmaps_in,
        tensor_maps_out=tmaps_out,
        tensors=TENSOR_FOLDER,
        batch_size=batch_size,
        valid_ratio=valid_ratio,
        test_ratio=test_ratio,
        test_modulo=0,
        num_workers=workers,
        cache_size=3.5e9 / workers,
        balance_csvs=[],
    )
    model = train_model_from_generators(
        model, generate_train, generate_valid, training_steps, validation_steps, batch_size,
        epochs, patience, OUTPUT_FOLDER, RECOVERY_MODEL_ID, True, True,
    )
    out_path = RECOVERY_MODEL_PATH
    return _predict_scalars_and_evaluate_from_generator(model, generate_test, tmaps_in, tmaps_out, test_steps, 'embed', out_path, 0)


if __name__ == '__main__':
    np.random.seed(SEED)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(FIGURE_FOLDER, exist_ok=True)
    now_string = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    load_config('INFO', OUTPUT_FOLDER, 'log_' + now_string, USER)
    if REMAKE_LABELS:
        logging.info('Remaking biosppy labels.')
        build_hr_biosppy_measurements_csv()
        plot_hr_from_biosppy_summary_stats()
    if RETRAIN_RECOVERY_MODEL:
        logging.info('Retraining recovery model.')
        train_recovery_model()
