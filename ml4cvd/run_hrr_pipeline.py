import os
import datetime
import numpy as np

from ml4cvd.logger import load_config
from ml4cvd.exercise_ecg_tensormaps import build_hr_biosppy_measurements_csv, plot_hr_from_biosppy_summary_stats, OUTPUT_FOLDER, USER, FIGURE_FOLDER


SEED = 217
REMAKE_LABELS = True


if __name__ == '__main__':
    np.random.seed(SEED)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(FIGURE_FOLDER, exist_ok=True)
    now_string = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    load_config('INFO', OUTPUT_FOLDER, 'log_' + now_string, USER)
    if REMAKE_LABELS:
        build_hr_biosppy_measurements_csv()
        plot_hr_from_biosppy_summary_stats()
