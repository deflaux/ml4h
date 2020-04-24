import datetime

from ml4cvd.logger import load_config
from ml4cvd.exercise_ecg_tensormaps import build_hr_biosppy_measurements_csv, plot_hr_from_biosppy_summary_stats, OUTPUT_FOLDER, USER


if __name__ == '__main__':
    now_string = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    load_config('INFO', OUTPUT_FOLDER, 'log_' + now_string, USER)
    build_hr_biosppy_measurements_csv()
    plot_hr_from_biosppy_summary_stats()
