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
from ml4cvd.metrics import coefficient_of_determination

from ml4cvd.logger import load_config
from ml4cvd.tensor_generators import test_train_valid_tensor_generators, TensorGenerator, BATCH_INPUT_INDEX, BATCH_OUTPUT_INDEX, BATCH_PATHS_INDEX
from ml4cvd.models import make_multimodal_multitask_model, BottleneckType, train_model_from_generators
from ml4cvd.exercise_ecg_tensormaps import OUTPUT_FOLDER, USER, FIGURE_FOLDER, BIOSPPY_MEASUREMENTS_PATH
from ml4cvd.exercise_ecg_tensormaps import RECOVERY_MODEL_PATH, TENSOR_FOLDER, RECOVERY_MODEL_ID, TEST_CSV, TEST_SET_LEN
from ml4cvd.exercise_ecg_tensormaps import RECOVERY_INFERENCE_FILE, HR_MEASUREMENT_TIMES, df_hr_col, df_hrr_col, df_diff_col
from ml4cvd.exercise_ecg_tensormaps import build_hr_biosppy_measurements_csv, plot_hr_from_biosppy_summary_stats, BIOSPPY_SENTINEL
from ml4cvd.exercise_ecg_tensormaps import ecg_bike_recovery_downsampled8x, _make_hr_biosppy_tmaps
from ml4cvd.exercise_ecg_tensormaps import plot_segment_prediction, DF_HR_COLS
from ml4cvd.defines import TENSOR_EXT
from ml4cvd.recipes import _make_tmap_nan_on_fail


SEED = 217
MAKE_LABELS = False or not os.path.exists(BIOSPPY_MEASUREMENTS_PATH)
TRAIN_RECOVERY_MODEL = False or not os.path.exists(RECOVERY_MODEL_PATH)
INFER_RECOVERY_MODEL = False or not os.path.exists(RECOVERY_INFERENCE_FILE)

RECOVERY_INPUT_TMAPS = [ecg_bike_recovery_downsampled8x]
RECOVERY_OUTPUT_TMAPS = sum(map(lambda x: list(x.values()), _make_hr_biosppy_tmaps()), [])
VALIDATION_RATIO = .1


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
    """trains model to get biosppy measurements from recovery"""
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


def _train_recovery_model():
    model = _get_recovery_model(False)
    batch_size = 128
    workers = cpu_count() * 2
    patience = 16
    epochs = 100
    data = pd.read_csv(BIOSPPY_MEASUREMENTS_PATH)
    error_ratio = len(data['error'].dropna()) / len(data)
    data_set_len = (len(data) - TEST_SET_LEN) * (1 - error_ratio) // batch_size  # approximation
    training_steps = int(data_set_len * (1 - VALIDATION_RATIO))
    validation_steps = int(data_set_len * VALIDATION_RATIO)

    generate_train, generate_valid, _ = test_train_valid_tensor_generators(
        tensor_maps_in=RECOVERY_INPUT_TMAPS,
        tensor_maps_out=RECOVERY_OUTPUT_TMAPS,
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
    try:
        train_model_from_generators(
            model, generate_train, generate_valid, training_steps, validation_steps, batch_size,
            epochs, patience, OUTPUT_FOLDER, RECOVERY_MODEL_ID, True, True,
        )
    finally:
        generate_train.kill_workers()
        generate_valid.kill_workers()


def _infer_recovery_model():
    """
    makes a csv of inference results
    """
    stats = Counter()
    tensor_paths_inferred = set()
    inference_tsv = RECOVERY_INFERENCE_FILE
    tensor_paths = [os.path.join(TENSOR_FOLDER, tp) for tp in sorted(os.listdir(TENSOR_FOLDER)) if os.path.splitext(tp)[-1].lower() == TENSOR_EXT]
    model = _get_recovery_model(use_model_file=True)
    no_fail_tmaps_out = [_make_tmap_nan_on_fail(tmap) for tmap in RECOVERY_OUTPUT_TMAPS]
    # hard code batch size to 1 so we can iterate over file names and generated tensors together in the tensor_paths for loop
    generate_test = TensorGenerator(
        1, RECOVERY_INPUT_TMAPS, no_fail_tmaps_out, tensor_paths, num_workers=0,
        cache_size=0, keep_paths=True, mixup=0,
    )
    generate_test.set_worker_paths(tensor_paths)
    with open(inference_tsv, mode='w') as inference_file:
        inference_writer = csv.writer(inference_file, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        header = ['sample_id']
        for otm in RECOVERY_OUTPUT_TMAPS:
            header.extend([otm.name+'_prediction', otm.name+'_actual'])
        inference_writer.writerow(header)

        while True:
            batch = next(generate_test)
            input_data, output_data, tensor_paths = batch[BATCH_INPUT_INDEX], batch[BATCH_OUTPUT_INDEX], batch[BATCH_PATHS_INDEX]
            if tensor_paths[0] in tensor_paths_inferred:
                next(generate_test)  # this prints end of epoch info
                logging.info(f"Inference on {stats['count']} tensors finished. Inference TSV file at: {inference_tsv}")
                break

            prediction = model.predict(input_data)
            if len(no_fail_tmaps_out) == 1:
                prediction = [prediction]

            csv_row = [os.path.basename(tensor_paths[0]).replace(TENSOR_EXT, '')]  # extract sample id
            for y, tm in zip(prediction, no_fail_tmaps_out):
                csv_row.append(str(tm.rescale(y)[0][0]))  # first index into batch then index into the 1x1 structure
                if ((tm.sentinel is not None and tm.sentinel == output_data[tm.output_name()][0][0])
                        or np.isnan(output_data[tm.output_name()][0][0])):
                    csv_row.append("NA")
                else:
                    csv_row.append(str(tm.rescale(output_data[tm.output_name()])[0][0]))

            inference_writer.writerow(csv_row)
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


def _evaluate_recovery_model():
    logging.info('Plotting recovery model results.')
    inference_results = pd.read_csv(RECOVERY_INFERENCE_FILE, sep='\t', dtype={'sample_id': str})
    test_ids = pd.read_csv(TEST_CSV, names=['sample_id'], dtype={'sample_id': str})
    test_results = inference_results.merge(test_ids, on='sample_id')

    # negative HRR measurements
    for t in HR_MEASUREMENT_TIMES[1:]:
        name = df_hrr_col(t) + '_prediction'
        col = test_results[name].dropna()
        logging.info(f'HRR_{t} had {(col < 0).mean() * 100:.2f}% negative predictions in hold out data.')
        logging.info(f'HRR_{t} had {(col < -5).mean() * 100:.2f}% predictions < -5 in hold out data.')

    # correlations with actual measurements
    ax_size = 5
    fig, axes = plt.subplots(2, len(HR_MEASUREMENT_TIMES), figsize=(ax_size * len(HR_MEASUREMENT_TIMES), 2 * ax_size))
    for i, t in enumerate(HR_MEASUREMENT_TIMES):
        name = df_hr_col(t)
        pred = test_results[name+'_prediction']
        actual = test_results[name+'_actual']
        not_na = ~np.isnan(pred) & ~np.isnan(actual) & (actual != BIOSPPY_SENTINEL)
        _scatter_plot(axes[0, i], actual[not_na], pred[not_na], f'HR at recovery time {t}')
    for i, t in enumerate(HR_MEASUREMENT_TIMES):
        if t == 0:
            continue
        name = df_hrr_col(t)
        pred = test_results[name+'_prediction']
        actual = test_results[name+'_actual']
        not_na = ~np.isnan(pred) & ~np.isnan(actual) & (actual != BIOSPPY_SENTINEL)
        _scatter_plot(axes[1, i], actual[not_na], pred[not_na], f'HRR at recovery time {t}')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_FOLDER, f'hr_recovery_measurements_model.png'))

    # distributions of predicted and actual measurements
    ax_size = 5
    fig, axes = plt.subplots(2, len(HR_MEASUREMENT_TIMES), figsize=(ax_size * len(HR_MEASUREMENT_TIMES), 2 * ax_size))
    for i, t in enumerate(HR_MEASUREMENT_TIMES):
        name = df_hr_col(t)
        pred = test_results[name+'_prediction']
        actual = test_results[name+'_actual']
        not_na = ~np.isnan(pred) & ~np.isnan(actual) & (actual != BIOSPPY_SENTINEL)
        _dist_plot(axes[0, i], actual[not_na], pred[not_na], f'HR at recovery time {t}')
    for i, t in enumerate(HR_MEASUREMENT_TIMES):
        if t == 0:
            continue
        name = df_hrr_col(t)
        pred = test_results[name+'_prediction']
        actual = test_results[name+'_actual']
        not_na = ~np.isnan(pred) & ~np.isnan(actual) & (actual != BIOSPPY_SENTINEL)
        _dist_plot(axes[1, i], actual[not_na], pred[not_na], f'HRR at recovery time {t}')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_FOLDER, f'hr_recovery_model_distributions.png'))

    # correlation of diffs vs. absolute error
    label_df = pd.read_csv(BIOSPPY_MEASUREMENTS_PATH, dtype={'sample_id': str}).merge(test_results, on='sample_id')
    fig, axes = plt.subplots(1, len(HR_MEASUREMENT_TIMES), figsize=(ax_size * len(HR_MEASUREMENT_TIMES), ax_size))
    for i, t in enumerate(HR_MEASUREMENT_TIMES):
        name = df_hr_col(t)
        diff_name = df_diff_col(t)
        pred = label_df[name+'_prediction']
        actual = label_df[name+'_actual']
        diff = label_df[diff_name]
        not_na = ~np.isnan(pred) & ~np.isnan(actual) & (actual != BIOSPPY_SENTINEL) & ~np.isnan(diff)
        mae = np.abs(pred - actual)
        label_df[name + '_mae'] = mae
        _scatter_plot(axes[i], mae[not_na], diff[not_na], f'HR vs. diff at recovery time {t}')
    plt.savefig(os.path.join(FIGURE_FOLDER, f'hr_recovery_model_error_vs_diffs.png'))

    # some really wrong predictions
    plots_per_time = 2
    plt.figure(figsize=(ax_size * 4, len(HR_MEASUREMENT_TIMES) * ax_size))
    for i, t in enumerate(HR_MEASUREMENT_TIMES):
        name = df_hr_col(t)
        mae = label_df[name + '_mae']
        select = (mae > np.quantile(mae.dropna(), .99)) & ~np.isnan(mae)
        rows = label_df[select].sample(2)
        for j, row in enumerate(rows.iterrows()):
            row = row[1]
            plt.subplot(len(HR_MEASUREMENT_TIMES), 2, 2 * i + 1 + j)
            plot_segment_prediction(
                row['sample_id'], t=t, pred=row[name+'_prediction'], actual=row[name+'_actual'],
                diff=row[df_diff_col(t)],
            )
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_FOLDER, f'hr_recovery_model_large_diff_segements.png'))


if __name__ == '__main__':
    """Always remakes figures"""
    np.random.seed(SEED)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(FIGURE_FOLDER, exist_ok=True)
    now_string = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    load_config('INFO', OUTPUT_FOLDER, 'log_' + now_string, USER)
    if MAKE_LABELS:
        logging.info('Making biosppy labels.')
        build_hr_biosppy_measurements_csv()
    plot_hr_from_biosppy_summary_stats()
    if TRAIN_RECOVERY_MODEL:
        logging.info('Training recovery model.')
        _train_recovery_model()
    if INFER_RECOVERY_MODEL:
        logging.info('Running inference on recovery model.')
        _infer_recovery_model()
    _evaluate_recovery_model()
    logging.info('Done.')
