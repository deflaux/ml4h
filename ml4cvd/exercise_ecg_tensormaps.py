import os
import time
import h5py
import biosppy
import seaborn as sns
import logging
import numpy as np
import pandas as pd
from typing import List, Union, Tuple, Dict
from itertools import combinations
from multiprocessing import Pool
import matplotlib.pyplot as plt

from ml4cvd.defines import TENSOR_EXT, MODEL_EXT
from ml4cvd.TensorMap import TensorMap, Interpretation, no_nans
from ml4cvd.tensor_writer_ukbb import tensor_path, first_dataset_at_path
from ml4cvd.normalizer import ZeroMeanStd1, Standardize
from ml4cvd.tensor_from_file import _get_tensor_at_first_date, _all_dates


PRETEST_DUR = 15  # DURs are measured in seconds
EXERCISE_DUR = 360
RECOVERY_DUR = 60
SAMPLING_RATE = 500
HR_MEASUREMENT_TIMES = 0, 10, 20, 30, 40, 50  # relative to recovery start
HR_SEGMENT_DUR = 10  # HR measurements in recovery coalesced across a segment of this length
TREND_TRACE_DUR_DIFF = 2  # Sum of phase durations from UKBB is 2s longer than the raw traces
LEAD_NAMES = 'lead_I', 'lead_2', 'lead_3'

TENSOR_FOLDER = '/mnt/disks/ecg-bike-tensors/2019-10-10/'
USER = 'ndiamant'
OUTPUT_FOLDER = f'/home/{USER}/ml/hrr_results'
COVARIATE_FILE = os.path.join(OUTPUT_FOLDER, 'covariates.tsv')
TEST_CSV = os.path.join(OUTPUT_FOLDER, 'test_ids.csv')
TEST_SET_LEN = 10000
BIOSPPY_MEASUREMENTS_FILE = os.path.join(OUTPUT_FOLDER, 'biosppy_hr_recovery_measurements.csv')
FIGURE_FOLDER = os.path.join(OUTPUT_FOLDER, 'figures')
RECOVERY_MODEL_ID = 'recovery_hr_model'
RECOVERY_MODEL_PATH = os.path.join(OUTPUT_FOLDER, RECOVERY_MODEL_ID, RECOVERY_MODEL_ID + MODEL_EXT)
RECOVERY_INFERENCE_FILE = os.path.join(OUTPUT_FOLDER, f'{RECOVERY_MODEL_ID}_inference.tsv')
PRETEST_LABEL_FILE = os.path.join(OUTPUT_FOLDER, f'hr_pretest_training_data.csv')
PRETEST_TRAINING_DUR = 10
BASELINE_MODEL_ID = 'pretest_baseline_model'
BASELINE_MODEL_PATH = os.path.join(OUTPUT_FOLDER, BASELINE_MODEL_ID, BASELINE_MODEL_ID + MODEL_EXT)
PRETEST_MODEL_ID = 'pretest_model'
PRETEST_MODEL_PATH = os.path.join(OUTPUT_FOLDER, PRETEST_MODEL_ID, PRETEST_MODEL_ID + MODEL_EXT)
HR_ACHIEVED_MODEL_ID = 'pretest_hr_achieved_model'
HR_ACHIEVED_MODEL_PATH = os.path.join(OUTPUT_FOLDER, HR_ACHIEVED_MODEL_ID, HR_ACHIEVED_MODEL_ID + MODEL_EXT)
PRETEST_INFERENCE_FILE = os.path.join(OUTPUT_FOLDER, 'pretest_model_inference.tsv')


# Tensor from file helpers
def _check_phase_full_len(hd5: h5py.File, phase: str):
    phase_len = _get_tensor_at_first_date(hd5, 'ecg_bike/continuous', f'{phase}_duration')
    valid = True
    if phase == 'pretest':
        valid &= phase_len == PRETEST_DUR
    elif phase == 'exercise':
        valid &= phase_len == EXERCISE_DUR
    elif phase == 'rest':
        valid &= phase_len == RECOVERY_DUR
    else:
        raise ValueError(f'Phase {phase} is not a valid phase.')
    if not valid:
        raise ValueError(f'{phase} phase is not full length.')


def _get_bike_ecg(hd5: h5py.File, start: int, stop: int, leads: Union[List[int], slice]):
    path_prefix, name = 'ecg_bike/float_array', 'full'
    ecg_dataset = first_dataset_at_path(hd5, tensor_path(path_prefix, name))
    tensor = np.array(ecg_dataset[start: stop, leads], dtype=np.float32)
    return tensor


def _get_downsampled_bike_ecg(length: int, hd5: h5py.File, start: int, rate: int, leads: Union[List[int], slice]):
    length = length * rate
    ecg = _get_bike_ecg(hd5, start, start + length, leads)
    ecg = _downsample_ecg(ecg, rate)
    return ecg


def _make_pretest_ecg_tff(downsample_rate: int, leads: Union[List[int], slice], random_start=True):
    def tff(tm: TensorMap, hd5: h5py.File, dependents=None):
        _check_phase_full_len(hd5, 'pretest')
        start = np.random.randint(0, SAMPLING_RATE * PRETEST_DUR - tm.shape[0] * downsample_rate) if random_start else 0
        return _get_downsampled_bike_ecg(tm.shape[0], hd5, start, downsample_rate, leads)
    return tff


def _get_trace_recovery_start(hd5: h5py.File) -> int:
    _check_phase_full_len(hd5, 'rest')
    pretest_dur = _get_tensor_at_first_date(hd5, 'ecg_bike/continuous', 'pretest_duration')
    exercise_dur = _get_tensor_at_first_date(hd5, 'ecg_bike/continuous', 'exercise_duration')
    return int(SAMPLING_RATE * (pretest_dur + exercise_dur - HR_SEGMENT_DUR / 2 - TREND_TRACE_DUR_DIFF))


def _make_recovery_ecg_tff(downsample_rate: int, leads: Union[List[int], slice], random_start=True):
    def tff(tm: TensorMap, hd5: h5py.File, dependents=None):
        shift = np.random.randint(-100, 0) if random_start else 0  # .2 second random shift as augmentation
        recovery_start = _get_trace_recovery_start(hd5) + shift
        return _get_downsampled_bike_ecg(tm.shape[0], hd5, recovery_start, downsample_rate, leads)
    return tff


# ECG transformations
def _warp_ecg(ecg):
    """Warning: does some weird stuff at the boundaries"""
    i = np.arange(ecg.shape[0])
    warped = i + (
        np.random.rand() * 100 * np.sin(i / (500 + np.random.rand() * 100))
        + np.random.rand() * 100 * np.cos(i / (500 + np.random.rand() * 100))
    )
    warped_ecg = np.zeros_like(ecg)
    for j in range(ecg.shape[1]):
        warped_ecg[:, j] = np.interp(i, warped, ecg[:, j])
    return warped_ecg


def _downsample_ecg(ecg, rate: int):
    """
    rate=2 halves the sampling rate. Uses linear interpolation. Requires ECG to be divisible by rate.
    """
    assert ecg.shape[0] % rate == 0  # TODO: make this not true so easier to hyperoptimize?
    i = np.arange(0, ecg.shape[0], rate)
    downsampled = np.zeros((ecg.shape[0] // rate, ecg.shape[1]))
    for j in range(ecg.shape[1]):
        downsampled[:, j] = np.interp(i, np.arange(ecg.shape[0]), ecg[:, j])
    return downsampled


def _rand_add_noise(ecg):
    noise_frac = np.random.rand() * .1  # max of 10% noise
    return ecg + noise_frac * ecg.mean(axis=0) * np.random.randn(*ecg.shape)


def _roll_ecg(ecg, shift: int):
    return np.roll(ecg, shift=shift, axis=0)


def _rand_roll_ecg(ecg):
    return _roll_ecg(ecg, shift=np.random.randint(ecg.shape[0]))


def _rand_offset_ecg(ecg):
    shift_frac = np.random.rand() * .01  # max % noise
    return ecg + shift_frac * ecg.mean(axis=0)


def _rand_scale_ecg(ecg):
    scale = 1 + np.random.randn() * .03
    return ecg * scale


# HR measurements from biosppy
BIOSPPY_DOWNSAMPLE_RATE = 4


def _get_segment_for_biosppy(ecg, mid_time: int):
    center = mid_time * SAMPLING_RATE // BIOSPPY_DOWNSAMPLE_RATE
    offset = (SAMPLING_RATE * HR_SEGMENT_DUR // BIOSPPY_DOWNSAMPLE_RATE) // 2
    return ecg[center - offset: center + offset]


def _get_biosppy_hr(segment: np.ndarray) -> float:
    return float(
        np.median(
            biosppy.signals.ecg.ecg(segment, sampling_rate=SAMPLING_RATE // BIOSPPY_DOWNSAMPLE_RATE, show=False)[-1],
        ),
    )


def _get_segments_for_biosppy(hd5: h5py.File):
    recovery_start_idx = _get_trace_recovery_start(hd5)
    length = (HR_MEASUREMENT_TIMES[-1] - HR_MEASUREMENT_TIMES[0] + HR_SEGMENT_DUR) * SAMPLING_RATE // BIOSPPY_DOWNSAMPLE_RATE
    ecg = _get_downsampled_bike_ecg(length, hd5, recovery_start_idx, BIOSPPY_DOWNSAMPLE_RATE, [0, 1, 2])
    for mid_time in HR_MEASUREMENT_TIMES:
        yield _get_segment_for_biosppy(ecg, mid_time + HR_SEGMENT_DUR // 2)


def _hr_and_diffs_from_segment(segment: np.ndarray) -> Tuple[float, float]:
    hr_per_lead = [_get_biosppy_hr(segment[:, i]) for i in range(segment.shape[-1])]
    max_diff = max(map(lambda pair: abs(pair[0] - pair[1]), combinations(hr_per_lead, 2)))
    return float(np.median(hr_per_lead)), max_diff


def _plot_segment(segment: np.ndarray):
    hr, max_diff = _hr_and_diffs_from_segment(segment)
    t = np.linspace(0, HR_SEGMENT_DUR, len(segment))
    for i, lead_name in enumerate(LEAD_NAMES):
        plt.plot(t, segment[:, i], label=lead_name)
    plt.xlabel('Time (s)')
    plt.legend()
    plt.title(f'hr: {hr:.2f}, max hr difference between leads: {max_diff:.2f}')


def plot_segment_prediction(sample_id: str, t: int, pred: float, actual: float, diff: float):
    t_idx = HR_MEASUREMENT_TIMES.index(t)
    with h5py.File(_path_from_sample_id(sample_id), 'r') as hd5:
        segment = list(_get_segments_for_biosppy(hd5))[t_idx]
        x = np.linspace(0, HR_SEGMENT_DUR, len(segment))
        for i, lead_name in enumerate(LEAD_NAMES):
            plt.title(
                '\n'.join([
                    f'{sample_id} at time {t} after recovery',
                    f'biosppy hr {actual:.2f}',
                    f'recovery model hr {pred:.2f}',
                    f'biosppy lead difference {diff:.2f}',
                ]),
            )
            plt.plot(x, segment[:, i], label=lead_name)


def _recovery_hrs_biosppy(hd5: h5py.File) -> List[Tuple[float, float]]:
    return list(map(_hr_and_diffs_from_segment, _get_segments_for_biosppy(hd5)))


def _path_from_sample_id(sample_id: str) -> str:
    return os.path.join(TENSOR_FOLDER, sample_id + TENSOR_EXT)


def _sample_id_from_hd5(hd5: h5py.File) -> int:
    return int(os.path.basename(hd5.filename).replace(TENSOR_EXT, ''))


def _plot_recovery_hrs(path: str):
    num_plots = len(HR_MEASUREMENT_TIMES)
    plt.figure(figsize=(10, 3 * num_plots))
    try:
        with h5py.File(path, 'r') as hd5:
            for i, segment in enumerate(_get_segments_for_biosppy(hd5)):
                plt.subplot(num_plots, 1, i + 1)
                _plot_segment(segment)
            plt.tight_layout()
            plt.savefig(os.path.join(FIGURE_FOLDER, f'biosppy_hr_recovery_measurements_{_sample_id_from_hd5(hd5)}.png'))
    except (ValueError, KeyError, OSError) as e:
        logging.debug(f'Plotting failed for {path} with error {e}.')


def df_hr_col(t):
    return f'{t}_hr'


def df_hrr_col(t):
    return f'{t}_hrr'


def df_diff_col(t):
    return f'{t}_diff'


DF_HR_COLS = [df_hr_col(t) for t in HR_MEASUREMENT_TIMES]
DF_DIFF_COLS = [df_diff_col(t) for t in HR_MEASUREMENT_TIMES]


def _recovery_hrs_from_path(path: str):
    sample_id = os.path.basename(path).replace(TENSOR_EXT, '')
    if sample_id.endswith('000'):
        logging.info(f'Processing sample_id {sample_id}.')
    hr_diff = np.full((len(HR_MEASUREMENT_TIMES), 2), np.nan)
    error = None
    try:
        with h5py.File(path, 'r') as hd5:
            hr_diff = np.array(_recovery_hrs_biosppy(hd5))
    except (ValueError, KeyError, OSError) as e:
        error = e
    measures = {'sample_id': sample_id, 'error': error}
    for i, (hr_col, diff_col) in enumerate(zip(DF_HR_COLS, DF_DIFF_COLS)):
        measures[hr_col] = hr_diff[i, 0]
        measures[diff_col] = hr_diff[i, 1]
    return measures


def plot_hr_from_biosppy_summary_stats():
    df = pd.read_csv(BIOSPPY_MEASUREMENTS_FILE)

    # HR summary stats
    plt.figure(figsize=(15, 7))
    for col, t in zip(DF_HR_COLS, HR_MEASUREMENT_TIMES):
        x = df[col].dropna()
        sns.distplot(x, label=f' Time = {t}\n mean = {x.mean():.2f}\n std = {x.std():.2f}\n top 5% = {np.quantile(x, .95):.2f}')
    plt.legend()
    plt.savefig(os.path.join(FIGURE_FOLDER, 'biosppy_hr_recovery_measurements_summary_stats.png'))

    # HR lead diff summary stats
    plt.figure(figsize=(15, 7))
    for col, t in zip(DF_DIFF_COLS, HR_MEASUREMENT_TIMES):
        x = df[col].dropna().copy()
        sns.distplot(x[x < 5], label=f' Time = {t}\n mean = {x.mean():.2f}\n std = {x.std():.2f}\n top 5% = {np.quantile(x, .95):.2f}')
    plt.legend()
    plt.savefig(os.path.join(FIGURE_FOLDER, 'biosppy_hr_diff_recovery_measurements_summary_stats.png'))

    # Random sample of hr trends
    plt.figure(figsize=(15, 7))
    trend_samples = df[DF_HR_COLS].sample(1000).values
    plt.plot(HR_MEASUREMENT_TIMES, (trend_samples - trend_samples[:, :1]).T, alpha=.2, linewidth=1, c='k')
    plt.axhline(0, c='k', linestyle='--')
    plt.savefig(os.path.join(FIGURE_FOLDER, 'biosppy_hr_trend_samples.png'))

    # correlation heat map
    plt.figure(figsize=(7, 7))
    sns.heatmap(df[DF_HR_COLS + DF_DIFF_COLS].corr(), annot=True, cbar=False)
    plt.savefig(os.path.join(FIGURE_FOLDER, 'biosppy_correlations.png'))
    plt.close()


def plot_pretest_label_summary_stats():
    df = pd.read_csv(PRETEST_LABEL_FILE)

    # HR summary stats
    plt.figure(figsize=(15, 7))
    for col, t in zip(DF_HR_COLS, HR_MEASUREMENT_TIMES):
        x = df[col].dropna()
        sns.distplot(x, label=f' Time = {t}\n mean = {x.mean():.2f}\n std = {x.std():.2f}\n top 5% = {np.quantile(x, .95):.2f}')
    plt.legend()
    plt.savefig(os.path.join(FIGURE_FOLDER, 'pretest_training_labels_summary_stats.png'))

    # Random sample of hr trends
    plt.figure(figsize=(15, 7))
    trend_samples = df[DF_HR_COLS].sample(1000).values
    plt.plot(HR_MEASUREMENT_TIMES, (trend_samples - trend_samples[:, :1]).T, alpha=.2, linewidth=1, c='k')
    plt.axhline(0, c='k', linestyle='--')
    plt.savefig(os.path.join(FIGURE_FOLDER, 'pretest_training_labels_hr_trend_samples.png'))

    # correlation heat map
    plt.figure(figsize=(7, 7))
    sns.heatmap(df[DF_HR_COLS].corr(), annot=True, cbar=False)
    plt.savefig(os.path.join(FIGURE_FOLDER, 'biosppy_correlations.png'))
    plt.close()


def build_hr_biosppy_measurements_csv():
    paths = [os.path.join(TENSOR_FOLDER, p) for p in sorted(os.listdir(TENSOR_FOLDER)) if p.endswith(TENSOR_EXT)]
    logging.info('Plotting 10 random hr measurements from biosppy.')
    for path in np.random.choice(paths, 10):
        _plot_recovery_hrs(path)
    pool = Pool()
    logging.info('Beginning to get hr measurements from biosppy.')
    now = time.time()
    measures = pool.map(_recovery_hrs_from_path, paths)
    df = pd.DataFrame(measures)
    delta_t = time.time() - now
    logging.info(f'Getting hr measurements from biosppy took {delta_t // 60} minutes at {delta_t / len(paths):.2f}s per path.')
    df.to_csv(BIOSPPY_MEASUREMENTS_FILE, index=False)


def build_pretest_training_labels():
    biosppy_labels = pd.read_csv(BIOSPPY_MEASUREMENTS_FILE, dtype={'sample_id': str})
    inferred_labels = pd.read_csv(RECOVERY_INFERENCE_FILE, dtype={'sample_id': str}, sep='\t')
    df = biosppy_labels.merge(inferred_labels, on='sample_id')
    new_df = pd.DataFrame()
    new_df['sample_id'] = df['sample_id']
    for t in HR_MEASUREMENT_TIMES:
        hr_name = df_hr_col(t)
        new_df[hr_name] = np.nanmean(df[[hr_name, time_to_pred_hr_col(t, RECOVERY_MODEL_ID)]], axis=1)  # Ensemble
        hrr_name = df_hrr_col(t)
        temp_name = 'temp_hrr'
        df[temp_name] = df[df_hr_col(HR_MEASUREMENT_TIMES[0])] - df[hr_name]
        new_df[hrr_name] = np.nanmean(df[[temp_name, time_to_pred_hrr_col(t, RECOVERY_MODEL_ID)]], axis=1)  # Ensemble
    new_df.to_csv(PRETEST_LABEL_FILE, index=False)


# Inference
ACTUAL_POSTFIX = '_actual'
PRED_POSTFIX = '_predicted'


def tmap_to_actual_col(tmap: TensorMap):
    return f'{tmap.name}{ACTUAL_POSTFIX}'


def tmap_to_pred_col(tmap: TensorMap, model_id: str):
    return f'{model_id}_{tmap.name}{PRED_POSTFIX}'


def time_to_pred_hr_col(t: int, model_id: str):
    return f'{model_id}_{df_hr_col(t)}{PRED_POSTFIX}'


def time_to_pred_hrr_col(t: int, model_id: str):
    return f'{model_id}_{df_hrr_col(t)}{PRED_POSTFIX}'


# Biosppy TensorMaps
BIOSPPY_SENTINEL = -1000
BIOSPPY_DIFF_CUTOFF = 5
HR_NORMALIZE = Standardize(100, 15)
HRR_NORMALIZE = Standardize(20, 10)


def _hr_file(file_name: str, t: int, hrr=False):
    error = None
    try:
        df = pd.read_csv(file_name)
        df = df.set_index('sample_id')
        if df_diff_col(HR_MEASUREMENT_TIMES[0]) not in df.columns:  # Hacky way to handle no diff case
            for t2 in HR_MEASUREMENT_TIMES:
                df[df_diff_col(t2)] = BIOSPPY_DIFF_CUTOFF - 1
    except FileNotFoundError as e:
        error = e

    def tensor_from_file(tm: TensorMap, hd5: h5py.File, dependents=None):
        if error:
            raise error
        sample_id = _sample_id_from_hd5(hd5)
        try:
            row = df.loc[sample_id]
            hr, diff = row[df_hr_col(t)], row[df_diff_col(t)]
            if diff > BIOSPPY_DIFF_CUTOFF:  # TODO: make diff analysis a validator
                return np.array([BIOSPPY_SENTINEL])
            if hrr:
                peak, peak_diff = row[df_hr_col(0)], row[df_diff_col(0)]
                if peak_diff > BIOSPPY_DIFF_CUTOFF:
                    return np.array([BIOSPPY_SENTINEL])
                out = peak - hr
            else:
                out = hr
            if np.isnan(out):
                out = BIOSPPY_SENTINEL
            return np.array([out])
        except KeyError:
            raise KeyError(f'Sample id not in {file_name}.')
    return tensor_from_file


def _make_hr_tmaps(file_name: str, parents=True) -> Tuple[Dict[int, TensorMap], Dict[int, TensorMap]]:
    biosppy_hr_tmaps = {}
    for t in HR_MEASUREMENT_TIMES:
        biosppy_hr_tmaps[t] = TensorMap(
            df_hr_col(t), shape=(1,), metrics=[],
            interpretation=Interpretation.CONTINUOUS,
            sentinel=HR_NORMALIZE.normalize(BIOSPPY_SENTINEL),
            tensor_from_file=_hr_file(file_name, t),
            normalization=HR_NORMALIZE,
        )
    biosppy_hrr_tmaps = {}
    for t in HR_MEASUREMENT_TIMES[1:]:
        biosppy_hrr_tmaps[t] = TensorMap(
            df_hrr_col(t), shape=(1,), metrics=[],
            interpretation=Interpretation.CONTINUOUS,
            sentinel=HRR_NORMALIZE.normalize(BIOSPPY_SENTINEL),
            tensor_from_file=_hr_file(file_name, t, hrr=True),
            parents=[biosppy_hr_tmaps[t], biosppy_hr_tmaps[HR_MEASUREMENT_TIMES[0]]] if parents else None,
            normalization=HRR_NORMALIZE,
        )
    return biosppy_hr_tmaps, biosppy_hrr_tmaps


# Covariates tensormaps
def _get_instance(hd5: h5py.File):
    path_prefix, name = 'ecg_bike/string', 'instance'
    dates = _all_dates(hd5, path_prefix, name)
    if not dates:
        raise ValueError(f'No {name} values values available.')
    return hd5[f'{tensor_path(path_prefix=path_prefix, name=name)}{min(dates)}/'][()]


def _make_covariate_tff(join_file: str, join_file_sep: str='\t'):
    df = None

    def tensor_from_file(tm: TensorMap, hd5: h5py.File, dependents=None):
        nonlocal df
        if df is None:
            df = pd.read_csv(COVARIATE_FILE, sep='\t')
            df = df.set_index('sample_id')
            to_merge = pd.read_csv(join_file,sep=join_file_sep).set_index('sample_id')
            df = df[df.index.isin(to_merge.index)]
        sample_id = _sample_id_from_hd5(hd5)
        try:
            row = df.loc[sample_id]
            instance = int(_get_instance(hd5))
            if type(row) == pd.Series:   # one instance case
                if row['instance'] != instance:
                    raise ValueError(f'No matching instance in covariate file.')
            else:  # many instance case
                row = row[row['instance'] == instance]
                if len(row) == 0:
                    raise ValueError(f'No matching instance in covariate file.')
            out = np.zeros(tm.shape, np.float32)
            val = row[tm.name]
            if tm.interpretation == Interpretation.CATEGORICAL:
                out[int(val)] = 1.
            else:
                out[0] = val
            return out
        except KeyError:
            raise KeyError(f'Sample id not covariate file.')
    return tensor_from_file


age = TensorMap(
    'age', shape=(1,),
    interpretation=Interpretation.CONTINUOUS,
    validator=no_nans, normalization=Standardize(57, 8),
    tensor_from_file=_make_covariate_tff(PRETEST_LABEL_FILE, ','),
)
bmi = TensorMap(
    'bmi', shape=(1,),
    interpretation=Interpretation.CONTINUOUS,
    validator=no_nans, normalization=Standardize(27, 5),
    tensor_from_file=_make_covariate_tff(PRETEST_LABEL_FILE, ','),
)
sex = TensorMap(
    'sex', shape=(2,), channel_map={'female': 1, 'male': 0},
    interpretation=Interpretation.CATEGORICAL,
    validator=no_nans, tensor_from_file=_make_covariate_tff(PRETEST_LABEL_FILE, ','),
)


# ECG TensorMaps
ecg_bike_recovery_downsampled8x = TensorMap(
    'recovery_ecg', shape=(3437, 3), interpretation=Interpretation.CONTINUOUS,
    validator=no_nans, normalization=Standardize(0, 100),
    tensor_from_file=_make_recovery_ecg_tff(8, [0, 1, 2]),
    cacheable=False, augmentations=[_rand_scale_ecg, _rand_add_noise, _rand_offset_ecg],
)


def make_pretest_tmap(downsample_rate: int, leads) -> TensorMap:
    assert PRETEST_TRAINING_DUR * SAMPLING_RATE % downsample_rate == 0
    return TensorMap(
        'pretest_ecg', shape=(PRETEST_TRAINING_DUR * SAMPLING_RATE // downsample_rate, len(leads)),
        interpretation=Interpretation.CONTINUOUS,
        validator=no_nans, normalization=Standardize(0, 100),
        tensor_from_file=_make_pretest_ecg_tff(downsample_rate, leads),
        cacheable=False, augmentations=[_rand_scale_ecg, _rand_add_noise, _rand_offset_ecg],
    )
