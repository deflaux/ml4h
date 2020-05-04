import os
import pandas as pd
import datetime
import numpy as np
from multiprocessing import cpu_count
import seaborn as sns
import logging
from collections import Counter
import csv
import matplotlib.pyplot as plt
from typing import Dict, Callable, List
from itertools import chain

from ml4cvd.logger import load_config
from ml4cvd.tensor_generators import test_train_valid_tensor_generators, TensorGenerator, BATCH_INPUT_INDEX, BATCH_OUTPUT_INDEX, BATCH_PATHS_INDEX
from ml4cvd.models import make_multimodal_multitask_model, BottleneckType, train_model_from_generators
from ml4cvd.exercise_ecg_tensormaps import OUTPUT_FOLDER, USER, FIGURE_FOLDER, BIOSPPY_MEASUREMENTS_FILE
from ml4cvd.exercise_ecg_tensormaps import RECOVERY_MODEL_PATH, TENSOR_FOLDER, RECOVERY_MODEL_ID, TEST_CSV, TEST_SET_LEN
from ml4cvd.exercise_ecg_tensormaps import RECOVERY_INFERENCE_FILE, HR_MEASUREMENT_TIMES, df_hr_col, df_hrr_col, df_diff_col
from ml4cvd.exercise_ecg_tensormaps import build_hr_biosppy_measurements_csv, plot_hr_from_biosppy_summary_stats, BIOSPPY_SENTINEL
from ml4cvd.exercise_ecg_tensormaps import ecg_bike_recovery_downsampled8x, _make_hr_tmaps, plot_pretest_label_summary_stats
from ml4cvd.exercise_ecg_tensormaps import plot_segment_prediction, build_pretest_training_labels, PRETEST_LABEL_FILE
from ml4cvd.exercise_ecg_tensormaps import make_pretest_tmap, PRETEST_MODEL_ID, PRETEST_MODEL_PATH
from ml4cvd.exercise_ecg_tensormaps import BASELINE_MODEL_ID, BASELINE_MODEL_PATH, PRETEST_INFERENCE_FILE
from ml4cvd.exercise_ecg_tensormaps import HR_ACHIEVED_MODEL_ID, HR_ACHIEVED_MODEL_PATH
from ml4cvd.exercise_ecg_tensormaps import age, sex, bmi, tmap_to_actual_col, tmap_to_pred_col, time_to_pred_hr_col, time_to_pred_hrr_col
from ml4cvd.exercise_ecg_tensormaps import time_to_actual_hr_col, time_to_actual_hrr_col
from ml4cvd.exercise_ecg_tensormaps import BIOSPPY_FIGURE_FOLDER, PRETEST_LABEL_FIGURE_FOLDER
from ml4cvd.defines import TENSOR_EXT
from ml4cvd.recipes import _make_tmap_nan_on_fail
from ml4cvd.metrics import coefficient_of_determination
from ml4cvd.TensorMap import TensorMap


SEED = 217
MAKE_LABELS = False or not os.path.exists(BIOSPPY_MEASUREMENTS_FILE)
TRAIN_RECOVERY_MODEL = False or not os.path.exists(RECOVERY_MODEL_PATH)
INFER_RECOVERY_MODEL = False or not os.path.exists(RECOVERY_INFERENCE_FILE)
MAKE_PRETEST_LABELS = False or not os.path.exists(PRETEST_LABEL_FILE)
TRAIN_BASELINE_MODEL = False or not os.path.exists(BASELINE_MODEL_PATH)
TRAIN_PRETEST_MODEL = False or not os.path.exists(PRETEST_MODEL_PATH)
TRAIN_HR_ACHIEVED_MODEL = False or not os.path.exists(HR_ACHIEVED_MODEL_PATH)
INFER_PRETEST_MODELS = (
    False
    or not os.path.exists(PRETEST_INFERENCE_FILE)
    or TRAIN_BASELINE_MODEL or TRAIN_PRETEST_MODEL or TRAIN_HR_ACHIEVED_MODEL
)

hr_tmaps, hrr_tmaps = _make_hr_tmaps(BIOSPPY_MEASUREMENTS_FILE)
RECOVERY_INPUT_TMAPS = [ecg_bike_recovery_downsampled8x]
RECOVERY_OUTPUT_TMAPS = list(hr_tmaps.values()) + list(hrr_tmaps.values())
VALIDATION_RATIO = .1

hr_tmaps, hrr_tmaps = _make_hr_tmaps(PRETEST_LABEL_FILE)
BASELINE_INPUT_TMAPS = [age, sex, bmi]
PRETEST_INPUT_TMAPS = [make_pretest_tmap(8, [0])] + BASELINE_INPUT_TMAPS  # TODO: later will come from hyperopt config
PRETEST_OUTPUT_TMAPS = list(hr_tmaps.values()) + list(hrr_tmaps.values())
hr_tmaps, hrr_tmaps = _make_hr_tmaps(PRETEST_LABEL_FILE, parents=False)
HR_ACHIEVED_INPUT_TMAPS = PRETEST_INPUT_TMAPS + [hr_tmaps[0]]  # TODO: later will come from hyperopt config
HR_ACHIEVED_OUTPUT_TMAPS = list(hr_tmaps.values())[1:] + list(hrr_tmaps.values())


def _get_results_from_bucket():
    """
    Gets trained models, test_csv, inference results, from bucket
    """
    pass


def _put_results_in_bucket():
    """
    Puts trained models, test_csv, inference results, from bucket
    """
    pass


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


def _infer_models(
        models: List[Callable], model_ids: List[str], inference_tsv: str,
        input_tmaps: List[TensorMap], output_tmaps: List[TensorMap],
):
    stats = Counter()
    tensor_paths_inferred = set()
    tensor_paths = [os.path.join(TENSOR_FOLDER, tp) for tp in sorted(os.listdir(TENSOR_FOLDER)) if os.path.splitext(tp)[-1].lower() == TENSOR_EXT]
    no_fail_tmaps_out = [_make_tmap_nan_on_fail(tmap) for tmap in output_tmaps]
    # hard code batch size to 1 so we can iterate over file names and generated tensors together in the tensor_paths for loop
    generate_test = TensorGenerator(
        1, input_tmaps, no_fail_tmaps_out, tensor_paths, num_workers=0,
        cache_size=0, keep_paths=True, mixup=0,
    )
    generate_test.set_worker_paths(tensor_paths)

    out_name_to_tmap = {tm.output_name(): tm for tm in output_tmaps}

    def model_output_to_tmap(out_name: str) -> TensorMap:
        return out_name_to_tmap[out_name]

    actual_cols = list(map(tmap_to_actual_col, output_tmaps))
    prediction_cols = sum(
        [
            [tmap_to_pred_col(model_output_to_tmap(out_name), m_id) for out_name in m.output_names]
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
            row = {}
            batch = next(generate_test)
            input_data, output_data, tensor_paths = batch[BATCH_INPUT_INDEX], batch[BATCH_OUTPUT_INDEX], batch[BATCH_PATHS_INDEX]
            if tensor_paths[0] in tensor_paths_inferred:
                next(generate_test)  # this prints end of epoch info
                logging.info(f"Inference on {stats['count']} tensors finished. Inference TSV file at: {inference_tsv}")
                break
            for m, m_id in zip(models, model_ids):
                for pred, out_name in zip(m.predict(input_data), m.output_names):
                    tm = model_output_to_tmap(out_name)
                    scaled = str(tm.rescale(pred[0][0]))
                    row[tmap_to_pred_col(tm, m_id)] = scaled
            row['sample_id'] = os.path.basename(tensor_paths[0]).replace(TENSOR_EXT, '')  # extract sample id
            for tm in no_fail_tmaps_out:
                if ((tm.sentinel is not None and tm.sentinel == output_data[tm.output_name()][0][0])
                        or np.isnan(output_data[tm.output_name()][0][0])):
                    row[tmap_to_actual_col(tm)] = 'NA'
                else:
                    row[tmap_to_actual_col(tm)] = (str(tm.rescale(output_data[tm.output_name()])[0][0]))

            inference_writer.writerow(row)
            tensor_paths_inferred.add(tensor_paths[0])
            stats['count'] += 1
            if stats['count'] % 250 == 0:
                logging.info(f"Wrote:{stats['count']} rows of inference.  Last tensor:{tensor_paths[0]}")


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

    # negative HRR measurements
    for t in HR_MEASUREMENT_TIMES[1:]:
        name = time_to_pred_hrr_col(t, m_id)
        col = test_results[name].dropna()
        logging.info(f'HRR_{t} had {(col < 0).mean() * 100:.2f}% negative predictions in hold out data.')
        logging.info(f'HRR_{t} had {(col < -5).mean() * 100:.2f}% predictions < -5 in hold out data.')

    # correlations with actual measurements
    ax_size = 5
    fig, axes = plt.subplots(2, len(HR_MEASUREMENT_TIMES), figsize=(ax_size * len(HR_MEASUREMENT_TIMES), 2 * ax_size))
    for i, t in enumerate(HR_MEASUREMENT_TIMES):
        pred_col = time_to_pred_hr_col(t, m_id)
        if pred_col not in test_results.columns:
            continue
        pred = test_results[pred_col]
        actual = test_results[time_to_actual_hr_col(t)]
        not_na = ~np.isnan(pred) & ~np.isnan(actual) & (actual != BIOSPPY_SENTINEL)
        _scatter_plot(axes[0, i], actual[not_na], pred[not_na], f'HR at recovery time {t}')
    for i, t in enumerate(HR_MEASUREMENT_TIMES):
        if t == 0:
            continue
        pred = test_results[time_to_pred_hrr_col(t, m_id)]
        actual = test_results[time_to_actual_hrr_col(t)]
        not_na = ~np.isnan(pred) & ~np.isnan(actual) & (actual != BIOSPPY_SENTINEL)
        _scatter_plot(axes[1, i], actual[not_na], pred[not_na], f'HRR at recovery time {t}')
    plt.tight_layout()
    plt.savefig(os.path.join(figure_folder, 'model_correlations.png'))

    # distributions of predicted and actual measurements
    ax_size = 5
    fig, axes = plt.subplots(2, len(HR_MEASUREMENT_TIMES), figsize=(ax_size * len(HR_MEASUREMENT_TIMES), 2 * ax_size))
    for i, t in enumerate(HR_MEASUREMENT_TIMES):
        pred_col = time_to_pred_hr_col(t, m_id)
        if pred_col not in test_results.columns:
            continue
        pred = test_results[pred_col]
        actual = test_results[time_to_actual_hr_col(t)]
        not_na = ~np.isnan(pred) & ~np.isnan(actual) & (actual != BIOSPPY_SENTINEL)
        _dist_plot(axes[0, i], actual[not_na], pred[not_na], f'HR at recovery time {t}')
    for i, t in enumerate(HR_MEASUREMENT_TIMES):
        if t == 0:
            continue
        pred = test_results[time_to_pred_hrr_col(t, m_id)]
        actual = test_results[time_to_actual_hrr_col(t)]
        not_na = ~np.isnan(pred) & ~np.isnan(actual) & (actual != BIOSPPY_SENTINEL)
        _dist_plot(axes[1, i], actual[not_na], pred[not_na], f'HRR at recovery time {t}')
    plt.tight_layout()
    plt.savefig(os.path.join(figure_folder, 'distributions.png'))

    # correlation of diffs vs. absolute error
    label_df = pd.read_csv(BIOSPPY_MEASUREMENTS_FILE, dtype={'sample_id': str}).merge(test_results, on='sample_id')
    fig, axes = plt.subplots(1, len(HR_MEASUREMENT_TIMES), figsize=(ax_size * len(HR_MEASUREMENT_TIMES), ax_size))
    for i, t in enumerate(HR_MEASUREMENT_TIMES):
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
    plt.figure(figsize=(ax_size * 4, len(HR_MEASUREMENT_TIMES) * ax_size))
    for i, t in enumerate(HR_MEASUREMENT_TIMES):
        name = df_hr_col(t)
        mae_name = name + '_mae'
        if mae_name not in label_df:
            continue
        mae = label_df[mae_name]
        select = (mae > np.quantile(mae.dropna(), .99)) & ~np.isnan(mae)
        rows = label_df[select].sample(2)
        for j, row in enumerate(rows.iterrows()):
            row = row[1]
            plt.subplot(len(HR_MEASUREMENT_TIMES), 2, 2 * i + 1 + j)
            plot_segment_prediction(
                row['sample_id'], t=t, pred=row[time_to_pred_hr_col(t, m_id)], actual=row[name+'_actual'],
                diff=row[df_diff_col(t)],
            )
    plt.tight_layout()
    plt.savefig(os.path.join(figure_folder, 'large_mae_segements.png'))


def _get_pretest_config():
    """Gets config of best hyperoptimized model"""
    pass  # TODO


def _get_pretest_baseline_model(use_model_file: bool):
    model = make_multimodal_multitask_model(
        tensor_maps_in=BASELINE_INPUT_TMAPS,
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
    config = _get_pretest_config()  # TODO: use config
    model = make_multimodal_multitask_model(
        tensor_maps_in=PRETEST_INPUT_TMAPS,
        tensor_maps_out=PRETEST_OUTPUT_TMAPS,
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
        model_file=PRETEST_MODEL_PATH if use_model_file else None,
    )
    return model


def _get_hr_achieved_model(use_model_file: bool):
    """builds model to get biosppy measurements from pretest + covariates"""
    config = _get_pretest_config()  # TODO: use config
    model = make_multimodal_multitask_model(
        tensor_maps_in=HR_ACHIEVED_INPUT_TMAPS,
        tensor_maps_out=HR_ACHIEVED_OUTPUT_TMAPS,
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
        model_file=HR_ACHIEVED_MODEL_PATH if use_model_file else None,
    )
    return model


def _demo_generator(gen: TensorGenerator):
    batch = next(gen)
    for k, v in chain(batch[BATCH_INPUT_INDEX].items(), batch[BATCH_INPUT_INDEX].items()):
        logging.info(f'\tKey {k} has mean {v.mean():.3f} and std {v.std():.3f}')


def _train_model(model, tmaps_in, tmaps_out, model_id, batch_size):
    """
    Eventually params should come from config generated by hyperoptimization
    """
    workers = cpu_count() * 2
    patience = 16
    epochs = 100
    data = pd.read_csv(PRETEST_LABEL_FILE)
    error_free_ratio = len(data[df_hr_col(0)].dropna()) / len(data)
    data_set_len = (len(data) - TEST_SET_LEN) * error_free_ratio // batch_size  # approximation
    training_steps = int(data_set_len * (1 - VALIDATION_RATIO))
    validation_steps = int(data_set_len * VALIDATION_RATIO)

    generate_train, generate_valid, _ = test_train_valid_tensor_generators(
        tensor_maps_in=tmaps_in,
        tensor_maps_out=tmaps_out,
        tensors=TENSOR_FOLDER,
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
        train_model_from_generators(
            model, generate_train, generate_valid, training_steps, validation_steps, batch_size,
            epochs, patience, OUTPUT_FOLDER, model_id, True, True,
        )
    finally:
        generate_train.kill_workers()
        generate_valid.kill_workers()


if __name__ == '__main__':
    """Always remakes figures"""
    np.random.seed(SEED)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(FIGURE_FOLDER, exist_ok=True)
    os.makedirs(BIOSPPY_FIGURE_FOLDER, exist_ok=True)
    os.makedirs(PRETEST_LABEL_FIGURE_FOLDER, exist_ok=True)
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
    if MAKE_PRETEST_LABELS:
        build_pretest_training_labels()
    plot_pretest_label_summary_stats()
    if TRAIN_BASELINE_MODEL:
        logging.info('Training baseline model.')
        _train_model(
            model=_get_pretest_baseline_model(False), tmaps_in=BASELINE_INPUT_TMAPS, tmaps_out=PRETEST_OUTPUT_TMAPS,
            model_id=BASELINE_MODEL_ID, batch_size=256,
        )
    if TRAIN_PRETEST_MODEL:
        logging.info('Training pretest model.')
        _train_model(
            model=_get_pretest_model(False), tmaps_in=PRETEST_INPUT_TMAPS, tmaps_out=PRETEST_OUTPUT_TMAPS,
            model_id=PRETEST_MODEL_ID, batch_size=256,
        )
    if TRAIN_HR_ACHIEVED_MODEL:
        logging.info('Training hr achieved model.')
        _train_model(
            model=_get_hr_achieved_model(False), tmaps_in=HR_ACHIEVED_INPUT_TMAPS, tmaps_out=HR_ACHIEVED_OUTPUT_TMAPS,
            model_id=HR_ACHIEVED_MODEL_ID, batch_size=256,
        )
    if INFER_PRETEST_MODELS:
        _infer_models(
            models=[_get_pretest_baseline_model(True), _get_pretest_model(True), _get_hr_achieved_model(True)],
            model_ids=[BASELINE_MODEL_ID, PRETEST_MODEL_ID, HR_ACHIEVED_MODEL_ID],
            input_tmaps=HR_ACHIEVED_INPUT_TMAPS,
            output_tmaps=PRETEST_OUTPUT_TMAPS,
            inference_tsv=PRETEST_INFERENCE_FILE,
        )
    _evaluate_model(BASELINE_MODEL_ID, PRETEST_INFERENCE_FILE)
    _evaluate_model(PRETEST_MODEL_ID, PRETEST_INFERENCE_FILE)
    _evaluate_model(HR_ACHIEVED_MODEL_ID, PRETEST_INFERENCE_FILE)
    logging.info('~~~~~~~~~~~~~~~~~~~ DONE ~~~~~~~~~~~~~~~~~~~')
