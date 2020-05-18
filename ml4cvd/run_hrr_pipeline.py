import os
import gc
import pandas as pd
import json
import datetime
import numpy as np
from multiprocessing import cpu_count
import seaborn as sns
import logging
import csv
import matplotlib.pyplot as plt
from typing import Dict, Callable, List, Tuple, Any, Optional, Set
from itertools import chain
import hyperopt
from hyperopt import hp, fmin, tpe
import pickle
import time
import tensorflow as tf

from ml4cvd.logger import load_config
from ml4cvd.hyperparameters import plot_trials, MAX_LOSS
from ml4cvd.tensor_generators import test_train_valid_tensor_generators, TensorGenerator, BATCH_INPUT_INDEX, BATCH_OUTPUT_INDEX, BATCH_PATHS_INDEX
from ml4cvd.models import make_multimodal_multitask_model, BottleneckType, train_model_from_generators
from ml4cvd.exercise_ecg_tensormaps import OUTPUT_FOLDER, USER, FIGURE_FOLDER, BIOSPPY_MEASUREMENTS_FILE
from ml4cvd.exercise_ecg_tensormaps import RECOVERY_MODEL_PATH, TENSOR_FOLDER, RECOVERY_MODEL_ID, TEST_CSV, TEST_SET_LEN
from ml4cvd.exercise_ecg_tensormaps import RECOVERY_INFERENCE_FILE, HR_MEASUREMENT_TIMES, df_hr_col, df_hrr_col, df_diff_col
from ml4cvd.exercise_ecg_tensormaps import build_hr_biosppy_measurements_csv, plot_hr_from_biosppy_summary_stats, BIOSPPY_SENTINEL
from ml4cvd.exercise_ecg_tensormaps import ecg_bike_recovery_downsampled8x, _make_hr_tmaps, plot_pretest_label_summary_stats
from ml4cvd.exercise_ecg_tensormaps import plot_segment_prediction, build_pretest_training_labels, PRETEST_LABEL_FILE
from ml4cvd.exercise_ecg_tensormaps import make_pretest_tmap, PRETEST_MODEL_ID, PRETEST_MODEL_PATH, PRETEST_75_ACHIEVED_INFERENCE_FILE
from ml4cvd.exercise_ecg_tensormaps import BASELINE_MODEL_ID, BASELINE_MODEL_PATH, PRETEST_INFERENCE_FILE
from ml4cvd.exercise_ecg_tensormaps import HR_ACHIEVED_MODEL_ID, HR_ACHIEVED_MODEL_PATH, bike_resting_hr, hr_achieved, hr_achieved_75
from ml4cvd.exercise_ecg_tensormaps import age, sex, bmi, tmap_to_actual_col, tmap_to_pred_col, time_to_pred_hr_col, time_to_pred_hrr_col
from ml4cvd.exercise_ecg_tensormaps import time_to_actual_hr_col, time_to_actual_hrr_col, HYPEROPT_BEST_FILE, make_rest_ids
from ml4cvd.exercise_ecg_tensormaps import BIOSPPY_FIGURE_FOLDER, PRETEST_LABEL_FIGURE_FOLDER, HYPEROPT_FIGURE_PATH
from ml4cvd.exercise_ecg_tensormaps import rest_age, rest_bmi, rest_sex, make_rest_ecg_tmap, rest_resting_hr
from ml4cvd.exercise_ecg_tensormaps import REST_MODEL_PATH, REST_HR_ACHIEVED_MODEL_PATH, REST_MODEL_ID
from ml4cvd.exercise_ecg_tensormaps import REST_HR_ACHIEVED_MODEL_ID, REST_TENSOR_FOLDER, REST_IDS, TRANSFER_INFERENCE_FILE, rest_hr_achieved
from ml4cvd.defines import TENSOR_EXT
from ml4cvd.recipes import _make_tmap_nan_on_fail
from ml4cvd.metrics import coefficient_of_determination
from ml4cvd.TensorMap import TensorMap


SEED = 217
MAKE_LABELS = False or not os.path.exists(BIOSPPY_MEASUREMENTS_FILE)
TRAIN_RECOVERY_MODEL = False or not os.path.exists(RECOVERY_MODEL_PATH)
INFER_RECOVERY_MODEL = False or not os.path.exists(RECOVERY_INFERENCE_FILE)
MAKE_PRETEST_LABELS = False or not os.path.exists(PRETEST_LABEL_FILE)
HYPEROPT_PRETEST_MODEL = False or not os.path.exists(HYPEROPT_BEST_FILE)
TRAIN_BASELINE_MODEL = False or not os.path.exists(BASELINE_MODEL_PATH)
TRAIN_PRETEST_MODEL = False or not os.path.exists(PRETEST_MODEL_PATH)
TRAIN_HR_ACHIEVED_MODEL = False or not os.path.exists(HR_ACHIEVED_MODEL_PATH)
INFER_PRETEST_MODELS = (
    False
    or not os.path.exists(PRETEST_INFERENCE_FILE)
    or TRAIN_BASELINE_MODEL or TRAIN_PRETEST_MODEL or TRAIN_HR_ACHIEVED_MODEL
)
INFER_75_ACHIEVED_MODEL = False or not os.path.exists(PRETEST_75_ACHIEVED_INFERENCE_FILE)
TRANSFER_PRETEST_MODEL = False or not os.path.exists(REST_MODEL_PATH)
TRANSFER_HR_ACHIEVED_MODEL = False or not os.path.exists(REST_HR_ACHIEVED_MODEL_PATH)
INFER_TRANSFER_MODELS = False or not os.path.exists(TRANSFER_INFERENCE_FILE)

HYPEROPT_MAX_TRIALS = 25
HISTORY_PATH = os.path.join(OUTPUT_FOLDER, 'hyperopt_histories')
TRIAL_PATH = os.path.join(HISTORY_PATH, 'trial_history.p')
hr_tmaps, hrr_tmaps = _make_hr_tmaps(BIOSPPY_MEASUREMENTS_FILE)
RECOVERY_INPUT_TMAPS = [ecg_bike_recovery_downsampled8x]
RECOVERY_OUTPUT_TMAPS = list(hr_tmaps.values()) + list(hrr_tmaps.values())
VALIDATION_RATIO = .1

hr_tmaps, hrr_tmaps = _make_hr_tmaps(PRETEST_LABEL_FILE)
PRETEST_COVARIATE_TMAPS = [age, sex, bmi]
BASELINE_INPUT_TMAPS = PRETEST_COVARIATE_TMAPS + [bike_resting_hr]
PRETEST_OUTPUT_TMAPS = [hr_tmaps[0], hr_tmaps[50], hrr_tmaps[50]]
hr_tmaps, hrr_tmaps = _make_hr_tmaps(PRETEST_LABEL_FILE, parents=False)
HR_ACHIEVED_OUTPUT_TMAPS = [hr_tmaps[50], hrr_tmaps[50]]
PRETEST_BOLT_FILE = os.path.join(OUTPUT_FOLDER, 'pretest_results_for_bolt.tsv')
PRETEST_TEST_BOLT_FILE = os.path.join(OUTPUT_FOLDER, 'pretest_test_results_for_bolt.tsv')

REST_COVARIATE_TMAPS = [rest_age, rest_sex, rest_bmi]
REST_BASELINE_INPUT_TMAPS = REST_COVARIATE_TMAPS + [rest_resting_hr]


def history_path():
    return os.path.join(HISTORY_PATH, f'history_{int(time.time())}.p')


def _get_recovery_model(use_model_file):
    """builds model to get biosppy measurements from recovery"""
    model = make_multimodal_multitask_model(
        tensor_maps_in=RECOVERY_INPUT_TMAPS,
        tensor_maps_out=RECOVERY_OUTPUT_TMAPS,
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
        model_file=RECOVERY_MODEL_PATH if use_model_file else None,
    )
    return model


def _handle_inference_batch(
        output_name_to_tmap: Dict[str, TensorMap], model, model_id: str, batch,
        visited_paths: Set[str], rows: List[Dict[str, str]],
):
    input_data, output_data, tensor_paths = batch[BATCH_INPUT_INDEX], batch[BATCH_OUTPUT_INDEX], batch[BATCH_PATHS_INDEX]
    pred = model.predict(input_data)
    for pred, out_name in zip(pred, model.output_names):
        tm = output_name_to_tmap[out_name]
        scaled = tm.rescale(pred)
        actual = output_data[tm.output_name()]
        for i, row in enumerate(rows):
            if tensor_paths[i] in visited_paths:
                continue
            visited_paths.add(tensor_paths[0])
            row[tmap_to_pred_col(tm, model_id)] = f'{scaled[i, 0]:.3f}'
            row['sample_id'] = os.path.basename(tensor_paths[i]).replace(TENSOR_EXT, '')  # extract sample id
            if ((tm.sentinel is not None and tm.sentinel == actual[i][0])
                    or np.isnan(actual[i][0])):
                row[tmap_to_actual_col(tm)] = 'NA'
            else:
                row[tmap_to_actual_col(tm)] = f'{tm.rescale(actual[i, 0]):.3f}'


def _infer_models(
        models: List[Callable], model_ids: List[str], inference_tsv: str,
        input_tmaps: List[TensorMap], output_tmaps: List[TensorMap], transfer: bool = False,
):
    tensor_folder = REST_TENSOR_FOLDER if transfer else TENSOR_FOLDER
    count = 0
    visited_paths = set()
    tensor_paths = [
        os.path.join(tensor_folder, tp) for tp in sorted(os.listdir(tensor_folder))
        if os.path.splitext(tp)[-1].lower() == TENSOR_EXT
    ]
    no_fail_tmaps_out = [_make_tmap_nan_on_fail(tmap) for tmap in output_tmaps]
    generate_test = None
    try:
        generate_test = TensorGenerator(
            128, input_tmaps, no_fail_tmaps_out, tensor_paths, num_workers=8,
            cache_size=0, keep_paths=True, mixup=0,
        )

        output_name_to_tmap = {tm.output_name(): tm for tm in output_tmaps}
        actual_cols = list(map(tmap_to_actual_col, no_fail_tmaps_out))
        prediction_cols = sum(
            [
                [tmap_to_pred_col(output_name_to_tmap[out_name], m_id) for out_name in m.output_names]
                for m, m_id in zip(models, model_ids)
            ],
            [],
        )
        with open(inference_tsv, mode='w') as inference_file:
            inference_writer = csv.DictWriter(
                inference_file, fieldnames=['sample_id'] + actual_cols + prediction_cols,
                delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL,
            )
            inference_writer.writeheader()
            while True:
                batch = next(generate_test)
                rows = [{} for _ in range(len(batch[BATCH_PATHS_INDEX]))]
                for model, model_id in zip(models, model_ids):
                    _handle_inference_batch(output_name_to_tmap, model, model_id, batch, visited_paths, rows)
                inference_writer.writerows(rows)
                if generate_test.stats_q.qsize() == generate_test.num_workers:
                    generate_test.aggregate_and_print_stats()
                    logging.info(f"Inference on {len(visited_paths)} tensors finished. Inference TSV file at: {inference_tsv}")
                    break
                count += 1
                logging.info(f"Wrote:{count} batches of inference.")
    finally:
        if generate_test:
            generate_test.kill_workers()


def _scatter_plot(ax, truth, prediction, title):
    ax.plot([np.min(truth), np.max(truth)], [np.min(truth), np.max(truth)], linewidth=2)
    ax.plot([np.min(prediction), np.max(prediction)], [np.min(prediction), np.max(prediction)], linewidth=4)
    pearson = np.corrcoef(prediction, truth)[1, 0]  # corrcoef returns full covariance matrix
    big_r_squared = coefficient_of_determination(truth, prediction)
    logging.info(f'{title} - pearson:{pearson:0.3f} r^2:{pearson*pearson:0.3f} R^2:{big_r_squared:0.3f}')
    ax.scatter(prediction, truth, label=f'Pearson:{pearson:0.3f} r^2:{pearson * pearson:0.3f} R^2:{big_r_squared:0.3f}', marker='.', s=1)
    ax.set_xlabel('Predictions')
    ax.set_ylabel('Actual')
    ax.set_title(title + '\n')
    ax.legend(loc="lower right")


def _dist_plot(ax, truth, prediction, title):
    ax.set_title(title)
    ax.legend(loc="lower right")
    sns.distplot(prediction, label='Predicted', color='r', ax=ax)
    sns.distplot(truth, label='Truth', color='b', ax=ax)
    ax.legend(loc="upper left")


def _evaluate_model(m_id: str, inference_file: str):
    logging.info('Plotting recovery model results.')
    inference_results = pd.read_csv(inference_file, sep='\t', dtype={'sample_id': str})
    test_ids = pd.read_csv(TEST_CSV, names=['sample_id'], dtype={'sample_id': str})
    test_results = inference_results.merge(test_ids, on='sample_id')
    figure_folder = os.path.join(FIGURE_FOLDER, f'{m_id}_results')
    os.makedirs(figure_folder, exist_ok=True)
    hrr_pred_times = [t for t in HR_MEASUREMENT_TIMES if time_to_pred_hrr_col(t, m_id) in inference_results.columns]
    hr_pred_times = [t for t in HR_MEASUREMENT_TIMES if time_to_pred_hr_col(t, m_id) in inference_results.columns]
    # negative HRR measurements
    for t in hrr_pred_times:
        name = time_to_pred_hrr_col(t, m_id)
        col = test_results[name].dropna()
        logging.info(f'HRR_{t} had {(col < 0).mean() * 100:.2f}% negative predictions in hold out data.')
        logging.info(f'HRR_{t} had {(col < -5).mean() * 100:.2f}% predictions < -5 in hold out data.')

    # correlations with actual measurements
    ax_size = 5
    fig, axes = plt.subplots(2, len(hr_pred_times), figsize=(ax_size * len(hr_pred_times), 2 * ax_size))
    if len(axes.shape) == 1:  # when len(hr_pred_times) == 1
        axes = axes[..., np.newaxis]
    for i, t in enumerate(hr_pred_times):
        pred_col = time_to_pred_hr_col(t, m_id)
        if pred_col not in test_results.columns:
            continue
        pred = test_results[pred_col]
        actual = test_results[time_to_actual_hr_col(t)]
        not_na = ~np.isnan(pred) & ~np.isnan(actual) & (actual != BIOSPPY_SENTINEL)
        _scatter_plot(axes[0, i], actual[not_na], pred[not_na], f'HR at recovery time {t}')
    for i, t in enumerate(hrr_pred_times):
        pred = test_results[time_to_pred_hrr_col(t, m_id)]
        actual = test_results[time_to_actual_hrr_col(t)]
        not_na = ~np.isnan(pred) & ~np.isnan(actual) & (actual != BIOSPPY_SENTINEL)
        _scatter_plot(axes[1, i], actual[not_na], pred[not_na], f'HRR at recovery time {t}')
    plt.tight_layout()
    plt.savefig(os.path.join(figure_folder, 'model_correlations.png'))

    # distributions of predicted and actual measurements
    ax_size = 5
    fig, axes = plt.subplots(2, len(hr_pred_times), figsize=(ax_size * len(hr_pred_times), 2 * ax_size))
    if len(axes.shape) == 1:  # when len(hr_pred_times) == 1
        axes = axes[..., np.newaxis]
    for i, t in enumerate(hr_pred_times):
        pred_col = time_to_pred_hr_col(t, m_id)
        if pred_col not in test_results.columns:
            continue
        pred = test_results[pred_col]
        actual = test_results[time_to_actual_hr_col(t)]
        not_na = ~np.isnan(pred) & ~np.isnan(actual) & (actual != BIOSPPY_SENTINEL)
        _dist_plot(axes[0, i], actual[not_na], pred[not_na], f'HR at recovery time {t}')
    for i, t in enumerate(hrr_pred_times):
        pred = test_results[time_to_pred_hrr_col(t, m_id)]
        actual = test_results[time_to_actual_hrr_col(t)]
        not_na = ~np.isnan(pred) & ~np.isnan(actual) & (actual != BIOSPPY_SENTINEL)
        _dist_plot(axes[1, i], actual[not_na], pred[not_na], f'HRR at recovery time {t}')
    plt.tight_layout()
    plt.savefig(os.path.join(figure_folder, 'distributions.png'))

    # correlation of diffs vs. absolute error
    label_df = pd.read_csv(BIOSPPY_MEASUREMENTS_FILE, dtype={'sample_id': str}).merge(test_results, on='sample_id')
    fig, axes = plt.subplots(1, len(hr_pred_times), figsize=(ax_size * len(hr_pred_times), ax_size))
    if type(axes) != np.ndarray:  # when len(hr_pred_times) == 1
        axes = [axes]
    for i, t in enumerate(hr_pred_times):
        name = df_hr_col(t)
        diff_name = df_diff_col(t)
        pred_col = time_to_pred_hr_col(t, m_id)
        if pred_col not in label_df.columns:
            continue
        pred = label_df[pred_col]
        actual = label_df[name+'_actual']
        diff = label_df[diff_name]
        not_na = ~np.isnan(pred) & ~np.isnan(actual) & (actual != BIOSPPY_SENTINEL) & ~np.isnan(diff)
        mae = np.abs(pred - actual)
        label_df[name + '_mae'] = mae
        _scatter_plot(axes[i], mae[not_na], diff[not_na], f'HR vs. diff at recovery time {t}')
    plt.savefig(os.path.join(figure_folder, 'model_error_vs_diffs.png'))

    # some really wrong predictions
    plots_per_time = 2
    plt.figure(figsize=(ax_size * 4, len(hr_pred_times) * ax_size))
    for i, t in enumerate(hr_pred_times):
        name = df_hr_col(t)
        mae_name = name + '_mae'
        if mae_name not in label_df:
            continue
        mae = label_df[mae_name]
        select = (mae > np.quantile(mae.dropna(), .99)) & ~np.isnan(mae)
        rows = label_df[select].sample(plots_per_time)
        for j, row in enumerate(rows.iterrows()):
            row = row[1]
            plt.subplot(len(hr_pred_times), 2, 2 * i + 1 + j)
            plot_segment_prediction(
                row['sample_id'], t=t, pred=row[time_to_pred_hr_col(t, m_id)], actual=row[name+'_actual'],
                diff=row[df_diff_col(t)],
            )
    plt.tight_layout()
    plt.savefig(os.path.join(figure_folder, 'large_mae_segements.png'))


def _get_baseline_model(use_model_file: bool, tmaps_in: List[TensorMap]):
    model = make_multimodal_multitask_model(
        tensor_maps_in=tmaps_in,
        tensor_maps_out=PRETEST_OUTPUT_TMAPS,
        activation='swish',
        learning_rate=1e-3,
        bottleneck_type=BottleneckType.FlattenRestructure,
        optimizer='radam',
        dense_layers=[64, 64],
        conv_layers=[32, 32, 32, 32, 32],  # lots of residual blocks with dilation
        dense_blocks=[32, 32, 32],
        model_file=BASELINE_MODEL_PATH if use_model_file else None,
    )
    return model


def _get_pretest_model(use_model_file: bool):
    """builds model to get biosppy measurements from pretest + covariates"""
    with open(HYPEROPT_BEST_FILE, 'r') as f:
        space = json.load(f)
    return _pretest_model_from_space(space, use_model_file)


def _get_hr_achieved_model(use_model_file: bool, constant_hr_achieved: bool = False):
    """builds model to get biosppy measurements from pretest + covariates"""
    with open(HYPEROPT_BEST_FILE, 'r') as f:
        space = json.load(f)
    return _hr_achieved_model_from_space(space, use_model_file, constant_hr_achieved=constant_hr_achieved)


def _demo_generator(gen: TensorGenerator):
    batch = next(gen)
    for k, v in chain(batch[BATCH_INPUT_INDEX].items(), batch[BATCH_INPUT_INDEX].items()):
        logging.info(f'\tKey {k} has mean {v.mean():.3f} and std {v.std():.3f}')


def _train_model(
        model, tmaps_in: List[TensorMap], tmaps_out: List[TensorMap], model_id: str, batch_size: int,
        transfer: bool = False,
) -> Tuple[Any, Dict]:
    workers = cpu_count() * 2
    patience = 8
    epochs = 100
    data = pd.read_csv(PRETEST_LABEL_FILE)
    test_ids = pd.read_csv(TEST_CSV, names=['sample_id'])
    test_ids = test_ids.set_index('sample_id')
    if transfer:
        rest_ids = pd.read_csv(REST_IDS)
        data = data.merge(rest_ids, on='sample_id')
    data = data[~data.index.isin(test_ids.index)]
    error_free_ratio = len(data[df_hr_col(0)].dropna()) / len(data)
    data_set_len = len(data) * error_free_ratio // batch_size  # approximation
    if data_set_len < 50:
        workers = 4
    training_steps = int(data_set_len * (1 - VALIDATION_RATIO))
    validation_steps = max(int(data_set_len * VALIDATION_RATIO), 1)

    generate_train, generate_valid, _ = test_train_valid_tensor_generators(
        tensor_maps_in=tmaps_in,
        tensor_maps_out=tmaps_out,
        tensors=REST_TENSOR_FOLDER if transfer else TENSOR_FOLDER,
        batch_size=batch_size,
        valid_ratio=VALIDATION_RATIO,
        test_ratio=.1,  # ignored, test comes from test_csv
        test_modulo=0,
        num_workers=workers,
        cache_size=3.5e9 / workers,
        balance_csvs=[],
        test_csv=TEST_CSV,
    )
    logging.info(f'Batch description for training {model_id}.')
    _demo_generator(generate_valid)
    try:
        model, history = train_model_from_generators(
            model, generate_train, generate_valid, training_steps, validation_steps, batch_size,
            epochs, patience, OUTPUT_FOLDER, model_id, True, True, return_history=True,
        )
    finally:
        generate_train.kill_workers()
        generate_valid.kill_workers()
    return model, history


def _pretest_model_from_space(space, use_model_file=False):
    pretest_tmap = make_pretest_tmap(downsample_rate=int(np.exp(space['log_downsampling'] / np.log(2))), leads=[0])
    tmaps_in = [pretest_tmap] + PRETEST_COVARIATE_TMAPS
    m = make_multimodal_multitask_model(
        tensor_maps_in=tmaps_in,
        tensor_maps_out=PRETEST_OUTPUT_TMAPS,
        activation='swish',
        learning_rate=1e-3,
        bottleneck_type=BottleneckType.FlattenRestructure,
        optimizer='radam',
        dense_layers=int(space['num_dense_layers']) * [int(space['dense_layer_units'])],
        dropout=0,
        conv_layers=int(space['num_res_blocks']) * [int(space['num_res_filters'])],
        dense_blocks=int(space['num_dense_blocks']) * [int(space['num_dense_block_filters'])],
        conv_type='conv',
        conv_normalize=space['conv_normalize'],
        conv_x=int(space['conv_x']),
        pool_type='max',
        pool_x=int(np.exp(space['log_pool_x']) / np.log(2)),
        conv_dilate=True,
        model_file=PRETEST_MODEL_PATH if use_model_file else None,
        block_size=3,
    )
    return m, tmaps_in


def _hr_achieved_model_from_space(space, use_model_file: bool = False, constant_hr_achieved: bool = False):
    pretest_tmap = make_pretest_tmap(downsample_rate=int(np.exp(space['log_downsampling'] / np.log(2))), leads=[0])
    tmaps_in = [pretest_tmap, hr_achieved_75 if constant_hr_achieved else hr_achieved] + PRETEST_COVARIATE_TMAPS
    m = make_multimodal_multitask_model(
        tensor_maps_in=tmaps_in,
        tensor_maps_out=HR_ACHIEVED_OUTPUT_TMAPS,
        activation='swish',
        learning_rate=1e-3,
        bottleneck_type=BottleneckType.FlattenRestructure,
        optimizer='radam',
        dense_layers=int(space['num_dense_layers']) * [int(space['dense_layer_units'])],
        dropout=0,
        conv_layers=int(space['num_res_blocks']) * [int(space['num_res_filters'])],
        dense_blocks=int(space['num_dense_blocks']) * [int(space['num_dense_block_filters'])],
        conv_type='conv',
        conv_normalize=space['conv_normalize'],
        conv_x=int(space['conv_x']),
        pool_type='max',
        pool_x=int(np.exp(space['log_pool_x']) / np.log(2)),
        conv_dilate=True,
        model_file=HR_ACHIEVED_MODEL_PATH if use_model_file else None,
        block_size=3,
    )
    return m, tmaps_in


def _rest_model_from_space(
        space, model_file: str, covariate_tmaps_in: List[TensorMap], tmaps_out: List[TensorMap],
        use_ecg: bool, transfer_model: bool,
):
    rest_ecg_tmap = make_rest_ecg_tmap(int(np.exp(space['log_downsampling'] / np.log(2))), {'strip_I': 0})
    tmaps_in = [rest_ecg_tmap] + covariate_tmaps_in if use_ecg else covariate_tmaps_in
    m = make_multimodal_multitask_model(
        tensor_maps_in=tmaps_in,
        tensor_maps_out=tmaps_out,
        activation='swish',
        learning_rate=1e-3,
        bottleneck_type=BottleneckType.FlattenRestructure,
        optimizer='radam',
        dense_layers=int(space['num_dense_layers']) * [int(space['dense_layer_units'])],
        dropout=0,
        conv_layers=int(space['num_res_blocks']) * [int(space['num_res_filters'])],
        dense_blocks=int(space['num_dense_blocks']) * [int(space['num_dense_block_filters'])],
        conv_type='conv',
        conv_normalize=space['conv_normalize'],
        conv_x=int(space['conv_x']),
        pool_type='max',
        pool_x=int(np.exp(space['log_pool_x']) / np.log(2)),
        conv_dilate=True,
        model_file=model_file if not transfer_model else None,
        block_size=3,
        model_layers=model_file if transfer_model else False,
        freeze_model_layers=True,
    )
    return m, tmaps_in


def _get_rest_model(transfer_model: bool, model_file: str):
    with open(HYPEROPT_BEST_FILE, 'r') as f:
        space = json.load(f)
    return _rest_model_from_space(
        space, model_file=model_file, covariate_tmaps_in=REST_COVARIATE_TMAPS, use_ecg=True,
        transfer_model=transfer_model, tmaps_out=PRETEST_OUTPUT_TMAPS,
    )


def _get_rest_hr_achieved_model(transfer_model: bool, model_file: str, constant_hr_achieved: bool):
    with open(HYPEROPT_BEST_FILE, 'r') as f:
        space = json.load(f)
    hr_achieved_tmap = hr_achieved_75 if constant_hr_achieved else rest_hr_achieved
    return _rest_model_from_space(
        space, model_file=model_file, covariate_tmaps_in=[hr_achieved_tmap] + REST_COVARIATE_TMAPS, use_ecg=True,
        transfer_model=transfer_model, tmaps_out=HR_ACHIEVED_OUTPUT_TMAPS,
    )


def optimize_hrr_model_architecture():
    conv_normalize = ['', 'batch_norm']
    space = {
        'conv_x': hp.quniform('conv_x', 2, 128, 1),
        'log_pool_x': hp.quniform('log_pool_x', 0, 3, 1),
        'log_downsampling': hp.quniform('log_downsampling', 0, 3, 1),
        'conv_normalize': hp.choice('conv_normalize', conv_normalize),
        'num_res_blocks': hp.quniform('num_res_blocks', 1, 6, 1),
        'num_res_filters': hp.quniform('num_res_filters', 8, 128, 1),
        'num_dense_blocks': hp.quniform('num_dense_blocks', 1, 6, 1),
        'num_dense_block_filters': hp.quniform('num_dense_block_filters', 8, 128, 1),
        'num_dense_layers': hp.quniform('num_dense_layers', 1, 6, 1),
        'dense_layer_units': hp.quniform('dense_layer_units', 8, 128, 1),
    }
    param_lists = {
        'conv_normalize': conv_normalize,
    }
    hyperparameter_optimizer(space, param_lists)


def hyperparameter_optimizer(space, param_lists=None):
    histories = []
    batch_size = 256
    if os.path.exists(TRIAL_PATH):
        with open(TRIAL_PATH, 'rb') as f:
            trials = pickle.load(f)
    else:
        trials = hyperopt.Trials()
    i = len(trials.trials)

    def loss_from_multimodal_multitask(x):
        model = None
        nonlocal i
        i += 1
        try:
            model, tmaps_in = _pretest_model_from_space(x)
            model, history = _train_model(model, tmaps_in, PRETEST_OUTPUT_TMAPS, 'hyperopt_model', batch_size)

            if model.count_params() > 9000000:
                logging.info(f"Model too big. Model has:{model.count_params()}. Return max loss.")
                return MAX_LOSS
            history.history['parameter_count'] = [model.count_params()]
            history.history['i'] = i
            history.history['space'] = x
            loss = np.median(sorted((history.history['val_loss']))[:5])  # median best val losses as proxy for test loss
            history.history['stated_loss'] = loss
            histories.append(history.history)
            logging.info(f"Iteration {i}: \nValidation Loss: {loss}")
            return loss
        except ValueError:
            logging.exception('ValueError trying to make a model for hyperparameter optimization. Returning max loss.')
            return MAX_LOSS
        except:
            logging.exception('Error trying hyperparameter optimization. Returning max loss.')
            return MAX_LOSS
        finally:
            del model
            gc.collect()
            plt.close('all')
    best = fmin(
        loss_from_multimodal_multitask, space=space, algo=tpe.suggest,
        max_evals=HYPEROPT_MAX_TRIALS + len(trials.trials), trials=trials,
    )
    with open(HYPEROPT_BEST_FILE, 'w') as f:
        json.dump(best, f)
    with open(TRIAL_PATH, 'wb') as f:
        pickle.dump(trials, f)
    with open(history_path(), 'wb') as f:
        pickle.dump(histories, f)


def plot_hyperopt():
    with open(TRIAL_PATH, 'rb') as f:
        trials = pickle.load(f)
    histories = []
    for path in os.listdir(HISTORY_PATH):
        if path.endswith('.p') and path.startswith('history'):
            with open(os.path.join(HISTORY_PATH, path), 'rb') as f:
                histories += pickle.load(f)
    histories_for_plot = [{'loss': [MAX_LOSS], 'val_loss': [MAX_LOSS], 'parameter_count': [1e7]} for _ in range(len(trials.trials))]
    for history in histories:
        histories_for_plot[history['i']] = history
    plot_trials(trials, histories_for_plot, HYPEROPT_FIGURE_PATH, {'conv_normalize': ['', 'batch_norm']})


def _get_hrr_cols(df: pd.DataFrame, t: int) -> List[str]:
    return [col for col in df.columns if df_hrr_col(t) in col and 'actual' not in col]


def prep_pretest_inferences_for_bolt():
    truth = pd.read_csv(PRETEST_LABEL_FILE)
    truth = truth[['sample_id'] + _get_hrr_cols(truth, 50)]
    pred = pd.read_csv(PRETEST_INFERENCE_FILE, sep='\t')
    pred = pred[['sample_id'] + _get_hrr_cols(pred, 50)]
    pred_75 = pd.read_csv(PRETEST_75_ACHIEVED_INFERENCE_FILE, sep='\t')
    pred_75_col = _get_hrr_cols(pred_75, 50)[0]
    pred_75 = pred_75[['sample_id', pred_75_col]]
    pred_75 = pred_75.rename(columns={pred_75_col: pred_75_col + '_75_hr_achieved'})
    combined = truth.merge(pred, on='sample_id')
    combined = combined.merge(pred_75, on='sample_id')
    combined.dropna()
    combined = combined.rename(columns={'sample_id': 'FID'})
    combined['IID'] = combined['FID']
    cols = list(combined.columns)
    # reorder so FID, IID are first
    cols.remove('FID')
    cols.remove('IID')
    combined = combined[['FID', 'IID'] + cols]
    # save tsvs
    combined.to_csv(PRETEST_BOLT_FILE, sep='\t', index=False)
    test_ids = pd.read_csv(TEST_CSV, names=['FID'])
    combined_test = combined.merge(test_ids, on='FID')
    combined_test.to_csv(PRETEST_TEST_BOLT_FILE, sep='\t', index=False)


if __name__ == '__main__':
    """Always remakes figures"""
    np.random.seed(SEED)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(FIGURE_FOLDER, exist_ok=True)
    os.makedirs(BIOSPPY_FIGURE_FOLDER, exist_ok=True)
    os.makedirs(PRETEST_LABEL_FIGURE_FOLDER, exist_ok=True)
    os.makedirs(HYPEROPT_FIGURE_PATH, exist_ok=True)
    os.makedirs(HISTORY_PATH, exist_ok=True)
    make_rest_ids()
    now_string = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    load_config('INFO', OUTPUT_FOLDER, 'log_' + now_string, USER)
    if MAKE_LABELS:
        logging.info('Making biosppy labels.')
        build_hr_biosppy_measurements_csv()
    plot_hr_from_biosppy_summary_stats()
    if TRAIN_RECOVERY_MODEL:
        logging.info('Training recovery model.')
        _train_model(
            model=_get_recovery_model(False), tmaps_in=RECOVERY_INPUT_TMAPS, tmaps_out=RECOVERY_OUTPUT_TMAPS, model_id=RECOVERY_MODEL_ID, batch_size=128,
        )
    if INFER_RECOVERY_MODEL:
        logging.info('Running inference on recovery model.')
        _infer_models(
            models=[_get_recovery_model(True)],
            model_ids=[RECOVERY_MODEL_ID],
            input_tmaps=RECOVERY_INPUT_TMAPS,
            output_tmaps=RECOVERY_OUTPUT_TMAPS,
            inference_tsv=RECOVERY_INFERENCE_FILE,
        )
    _evaluate_model(RECOVERY_MODEL_ID, RECOVERY_INFERENCE_FILE)
    plt.close('all')
    if MAKE_PRETEST_LABELS:
        build_pretest_training_labels()
    plot_pretest_label_summary_stats()
    if HYPEROPT_PRETEST_MODEL:
        logging.info('Hyperoptimizing pretest model.')
        optimize_hrr_model_architecture()
    plot_hyperopt()
    if TRAIN_BASELINE_MODEL:
        tf.keras.backend.clear_session()
        logging.info('Training baseline model.')
        _train_model(
            model=_get_baseline_model(False, BASELINE_INPUT_TMAPS),
            tmaps_in=BASELINE_INPUT_TMAPS, tmaps_out=PRETEST_OUTPUT_TMAPS,
            model_id=BASELINE_MODEL_ID, batch_size=256,
        )
    if TRAIN_PRETEST_MODEL:
        tf.keras.backend.clear_session()
        logging.info('Training pretest model.')
        model, tmaps_in = _get_pretest_model(False)
        _train_model(
            model=model, tmaps_in=tmaps_in, tmaps_out=PRETEST_OUTPUT_TMAPS,
            model_id=PRETEST_MODEL_ID, batch_size=256,
        )
    if TRAIN_HR_ACHIEVED_MODEL:
        tf.keras.backend.clear_session()
        logging.info('Training hr achieved model.')
        model, hr_achieved_tmaps_in = _get_hr_achieved_model(False)
        _train_model(
            model=model, tmaps_in=hr_achieved_tmaps_in, tmaps_out=HR_ACHIEVED_OUTPUT_TMAPS,
            model_id=HR_ACHIEVED_MODEL_ID, batch_size=256,
        )
    if INFER_PRETEST_MODELS:
        logging.info('Running inference on pretest models.')
        _, hr_achieved_tmaps_in = _get_hr_achieved_model(False)
        _infer_models(
            models=[_get_baseline_model(True, BASELINE_INPUT_TMAPS), _get_pretest_model(True)[0], _get_hr_achieved_model(True)[0]],
            model_ids=[BASELINE_MODEL_ID, PRETEST_MODEL_ID, HR_ACHIEVED_MODEL_ID],
            input_tmaps=hr_achieved_tmaps_in + [bike_resting_hr],
            output_tmaps=PRETEST_OUTPUT_TMAPS,
            inference_tsv=PRETEST_INFERENCE_FILE,
        )
    if INFER_75_ACHIEVED_MODEL:
        hr_achieved_model, hr_achieved_tmaps_in = _get_hr_achieved_model(use_model_file=True, constant_hr_achieved=True)
        _infer_models(
            models=[hr_achieved_model],
            model_ids=[HR_ACHIEVED_MODEL_ID],
            input_tmaps=hr_achieved_tmaps_in,
            output_tmaps=HR_ACHIEVED_OUTPUT_TMAPS,
            inference_tsv=PRETEST_75_ACHIEVED_INFERENCE_FILE,
        )
    logging.info('Evaluating pretest model predictions.')
    _evaluate_model(BASELINE_MODEL_ID, PRETEST_INFERENCE_FILE)
    _evaluate_model(PRETEST_MODEL_ID, PRETEST_INFERENCE_FILE)
    _evaluate_model(HR_ACHIEVED_MODEL_ID, PRETEST_INFERENCE_FILE)
    prep_pretest_inferences_for_bolt()
    plt.close('all')
    if TRANSFER_PRETEST_MODEL:
        tf.keras.backend.clear_session()
        logging.info('Transferring pretest model.')
        pretest_transfer, tmaps_in = _get_rest_model(transfer_model=True, model_file=PRETEST_MODEL_PATH)
        _train_model(
            model=pretest_transfer, tmaps_in=tmaps_in, tmaps_out=PRETEST_OUTPUT_TMAPS,
            model_id=REST_MODEL_ID, batch_size=256, transfer=True,
        )
    if TRANSFER_HR_ACHIEVED_MODEL:
        tf.keras.backend.clear_session()
        logging.info('Transferring hr achieved model.')
        hr_achieved_transfer, tmaps_in = _get_rest_hr_achieved_model(
            transfer_model=True, model_file=HR_ACHIEVED_MODEL_PATH, constant_hr_achieved=False,
        )
        _train_model(
            model=hr_achieved_transfer, tmaps_in=tmaps_in, tmaps_out=HR_ACHIEVED_OUTPUT_TMAPS,
            model_id=REST_HR_ACHIEVED_MODEL_ID, batch_size=256, transfer=True,
        )
    if INFER_TRANSFER_MODELS:
        logging.info('Inferring transferred models.')
        pretest_transfer, _ = _get_rest_model(transfer_model=False, model_file=REST_MODEL_PATH)
        hr_achieved_transfer, hr_achieved_tmaps_in = _get_rest_hr_achieved_model(
            transfer_model=False, model_file=REST_HR_ACHIEVED_MODEL_PATH, constant_hr_achieved=True,
        )
        baseline_rest = _get_baseline_model(use_model_file=True, tmaps_in=REST_BASELINE_INPUT_TMAPS)
        _infer_models(
            models=[baseline_rest, pretest_transfer, hr_achieved_transfer],
            model_ids=[BASELINE_MODEL_ID, REST_MODEL_ID, REST_HR_ACHIEVED_MODEL_ID],
            input_tmaps=hr_achieved_tmaps_in + [rest_resting_hr],
            output_tmaps=PRETEST_OUTPUT_TMAPS,
            inference_tsv=TRANSFER_INFERENCE_FILE,
            transfer=True,
        )
    logging.info('Evaluating transferred model predictions.')
    _evaluate_model(BASELINE_MODEL_ID, TRANSFER_INFERENCE_FILE)
    _evaluate_model(REST_MODEL_ID, TRANSFER_INFERENCE_FILE)
    _evaluate_model(REST_HR_ACHIEVED_MODEL_ID, TRANSFER_INFERENCE_FILE)

    logging.info('~~~~~~~~~~~~~~~~~~~ DONE ~~~~~~~~~~~~~~~~~~~')
