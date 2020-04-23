import os
import time
import h5py
import biosppy
import pathlib
import logging
import numpy as np
import pandas as pd
from typing import List, Union, Tuple
from itertools import combinations
from multiprocessing import Pool
import matplotlib.pyplot as plt

from ml4cvd.metrics import pearson
from ml4cvd.TensorMap import TensorMap, Interpretation
from ml4cvd.tensor_writer_ukbb import tensor_path, first_dataset_at_path
from ml4cvd.tensor_from_file import _get_tensor_at_first_date


PRETEST_DUR = 15  # DURs are measured in seconds
EXERCISE_DUR = 360
RECOVERY_DUR = 60
SAMPLING_RATE = 500
HR_MEASUREMENT_TIMES = 0, 10, 20, 30, 40, 50  # relative to recovery start
HR_SEGMENT_DUR = 10  # HR measurements in recovery coalesced across a segment of this length
TREND_TRACE_DUR_DIFF = 2  # Sum of phase durations from UKBB is 2s longer than the raw traces

TENSOR_FOLDER = '/mnt/disks/ecg-bike-tensors/2019-10-10/'
USER = 'ndiamant'
OUTPUT_FOLDER = f'/home/{USER}/ml/hrr_results'
BIOSPPY_MEASUREMENTS_PATH = os.path.join(OUTPUT_FOLDER, 'biosppy_hr_recovery_measurements.csv')
FIGURE_FOLDER = os.path.join(OUTPUT_FOLDER, 'figures')


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


def _make_recovery_ecg_tff(downsample_rate: int, leads: Union[List[int], slice], random_start=False):
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
    shift_frac = np.random.rand() * .1  # max of 10% noise
    return ecg + shift_frac * ecg.mean(axis=0)


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
    plt.plot(segment)
    plt.title(f'hr: {hr:.2f}, max hr difference between leads: {max_diff:.2f}')


def _recovery_hrs_biosppy(hd5: h5py.File) -> List[Tuple[float, float]]:
    return list(map(_hr_and_diffs_from_segment, _get_segments_for_biosppy(hd5)))


def _sample_id_from_hd5(hd5: h5py.File):
    return os.path.basename(hd5.filename).replace('.hd5', '')


def _plot_recovery_hrs(hd5: h5py.File):
    num_plots = len(HR_MEASUREMENT_TIMES)
    plt.figure(figsize=(10, 3 * num_plots))
    for i, segment in enumerate(_get_segments_for_biosppy(hd5)):
        plt.subplot(num_plots, 1, i + 1)
    plt.savefig(os.path.join(FIGURE_FOLDER, f'biosppy_hr_recovery_measurements_{_sample_id_from_hd5(hd5)}.png'))


def _recovery_hrs_from_path(path: str):
    sample_id = os.path.basename(path).replace('.hd5', '')
    hr_diff = np.full((len(HR_MEASUREMENT_TIMES), 2), np.nan)
    error = None
    try:
        with h5py.File(path, 'r') as hd5:
            hr_diff = np.array(_recovery_hrs_biosppy(hd5))
    except (ValueError, KeyError, OSError) as e:
        error = e
    measures = {'sample_id': sample_id, 'error': error}
    for i, t in enumerate(HR_MEASUREMENT_TIMES):
        measures[f'{t}_hr'] = hr_diff[i, 0]
        measures[f'{t}_diff'] = hr_diff[i, 1]
    return measures


def build_hr_biosppy_measurements_csv():
    paths = [os.path.join(TENSOR_FOLDER, p) for p in os.listdir(TENSOR_FOLDER)]
    pool = Pool()
    now = time.time()
    measures = pool.map(_recovery_hrs_from_path, paths)
    df = pd.DataFrame(measures)
    delta_t = time.time() - now
    logging.info(f'Getting hr measurements from biosppy took {delta_t // 60} minutes at {delta_t / len(paths):.2f}s per path.')
    print(f'Getting hr measurements from biosppy took {delta_t // 60} minutes at {delta_t / len(paths):.2f}s per path.')
    df.to_csv(BIOSPPY_MEASUREMENTS_PATH, index=False)


# ECG TensorMaps
TMAPS = {}
BIOSPPY_SENTINEL = -1000
BIOSPPY_DIFF_CUTOFF = 5


def _hr_biosppy_file(file_name: str, time: int, hrr=False):
    error = None
    try:
        df = pd.read_csv(file_name, dtype={'sample_id': str})
        df = df.set_index('sample_id')
    except FileNotFoundError as e:
        error = e

    def tensor_from_file(tm: TensorMap, hd5: h5py.File, dependents=None):
        if error:
            raise error
        sample_id = _sample_id_from_hd5(hd5)
        try:
            row = df.loc[sample_id]
            hr, diff = row[f'{time}_hr'], row[f'{time}_diff']
            if diff > BIOSPPY_DIFF_CUTOFF:
                return np.array([BIOSPPY_SENTINEL])
            if hrr:
                peak, peak_diff = row[f'0_hr'], row[f'0_diff']
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


def _make_hr_biosppy_tmaps():
    for time in HR_MEASUREMENT_TIMES:
        TMAPS[f'ecg-bike-{time}_hr'] = TensorMap(
            f'{time}_hr', metrics=['mae', pearson], shape=(1,),
            interpretation=Interpretation.CONTINUOUS,
            sentinel=BIOSPPY_SENTINEL,
            tensor_from_file=_hr_biosppy_file(BIOSPPY_MEASUREMENTS_PATH, time),
        )
    for time in HR_MEASUREMENT_TIMES:
        TMAPS[f'ecg-bike-{time}_hrr'] = TensorMap(
            f'{time}_hrr', metrics=['mae', pearson], shape=(1,),
            interpretation=Interpretation.CONTINUOUS,
            sentinel=BIOSPPY_SENTINEL,
            tensor_from_file=_hr_biosppy_file(BIOSPPY_MEASUREMENTS_PATH, time, hrr=True),
            parents=[TMAPS[f'ecg-bike-{time}_hr'], TMAPS['ecg-bike-0_hr']],
        )
