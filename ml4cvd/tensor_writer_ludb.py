# tensor_writer_ukbb.py
#
# UK Biobank-specific tensor writing, SQL querying, data munging goes here
#

# Imports
import os
import h5py
import wfdb
import logging
import datetime
import operator
import numpy as np
from typing import Dict, List, Tuple
from timeit import default_timer as timer
from collections import Counter, defaultdict

import matplotlib
matplotlib.use('Agg')  # Need this to write images from the GSA servers.  Order matters:
import matplotlib.pyplot as plt  # First import matplotlib, then use Agg, then import plt

from ml4cvd.plots import plot_value_counter, plot_histograms
from ml4cvd.defines import DataSetType, dataset_name_from_meaning
from ml4cvd.defines import TENSOR_EXT, JOIN_CHAR, CONCAT_CHAR, HD5_GROUP_CHAR, DATE_FORMAT

MISSING_DATE = datetime.date(year=1900, month=1, day=1)
ECG_SINUS = ['Normal_sinus_rhythm', 'Sinus_bradycardia', 'Marked_sinus_bradycardia', 'Atrial_fibrillation']
ECG_TAGS_TO_WRITE = ['VentricularRate', 'PQInterval', 'PDuration', 'QRSDuration', 'QTInterval', 'QTCInterval', 'RRInterval', 'PPInterval',
                     'SokolovLVHIndex', 'PAxis', 'RAxis', 'TAxis', 'QTDispersion', 'QTDispersionBazett', 'QRSNum', 'POnset', 'POffset', 'QOnset',
                     'QOffset', 'TOffset']
SECONDS_PER_MINUTE = 60


def write_tensors_ludb(a_id: str,
                       xml_folder: str,
                       output_folder: str,
                       tensors: str,
                       min_sample_id: int,
                       max_sample_id: int) -> None:
    """Write tensors as HD5 files containing ECGs and segmentation from the LUDB dataset

    One HD5 file is generated per sample.

    :param a_id: User chosen string to identify this run
    :param xml_folder: Path to folder containing ECG XML files
    :param output_folder: Folder to write outputs to (mostly for debugging)
    :param tensors: Folder to populate with HD5 tensors
    :param min_sample_id: Minimum sample id to generate, for parallelization
    :param max_sample_id: Maximum sample id to generate, for parallelization

    :return: None
    """
    stats = Counter()
    continuous_stats = defaultdict(list)
    sample_ids = range(min_sample_id, max_sample_id)
    for sample_id in sorted(sample_ids):

        start_time = timer()  # Keep track of elapsed execution time

        tensor_path = os.path.join(tensors, str(sample_id) + TENSOR_EXT)
        if not os.path.exists(os.path.dirname(tensor_path)):
            os.makedirs(os.path.dirname(tensor_path))
        try:
            with h5py.File(tensor_path, 'w') as hd5:
                _write_tensors_from_wfdb(xml_folder, hd5, sample_id, stats)
                stats['Tensors written'] += 1
        except AttributeError:
            logging.exception('Encountered AttributeError trying to write a LUDB tensor at path:{}'.format(tensor_path))
            logging.info('Deleting attempted tensor at path:{}'.format(tensor_path))
            os.remove(tensor_path)
        except ValueError:
            logging.exception('Encountered ValueError trying to write a LUDB tensor at path:{}'.format(tensor_path))
            logging.info('Deleting attempted tensor at path:{}'.format(tensor_path))
            os.remove(tensor_path)
        except RuntimeError:
            logging.exception('Encountered RuntimeError trying to write a LUDB tensor at path:{}'.format(tensor_path))
            logging.info('Deleting attempted tensor at path:{}'.format(tensor_path))
            os.remove(tensor_path)
        except IndexError:
            logging.exception('Encountered IndexError trying to write a LUDB tensor at path:{}'.format(tensor_path))
            logging.info('Deleting attempted tensor at path:{}'.format(tensor_path))
            os.remove(tensor_path)
        except OSError:
            logging.exception('Encountered OSError trying to write a LUDB tensor at path:{}'.format(tensor_path))
            logging.info('Deleting attempted tensor at path:{}'.format(tensor_path))
            os.remove(tensor_path)

        end_time = timer()
        elapsed_time = end_time - start_time
        logging.info("Populated {} in {} seconds.".format(tensor_path, elapsed_time))


def _write_tensors_from_wfdb(xml_folder, hd5, sample_id, stats):
    xmlp = xml_folder + str(sample_id)
    rest_group = 'ecg_rest' + HD5_GROUP_CHAR
    annotation_group = 'ecg_rest_annotation' + HD5_GROUP_CHAR
    ann_ext = 'atr'

    record = wfdb.rdrecord(xmlp)
    logging.info('Got ECG for sample:{}'.format(sample_id))
    diagnosis_comments = record.comments
    for i, comment in enumerate(diagnosis_comments):
        if '<age>' in comment:
            hd5.create_dataset('continuous' + HD5_GROUP_CHAR + 'age', data=int(comment.split()[-1]), dtype=np.int)
        if '<sex>' in comment:
            hd5.create_dataset('categorical' + HD5_GROUP_CHAR + 'sex', data=str(comment.split()[-1]), dtype=np.str)
        if '<diagnoses>' in comment:
            if 'ecg_rest_text' in hd5:
                continue
            diagnosis_text = []
            for d in diagnosis_comments[i:]:
                diagnosis_text.append(d.text.replace(',', '').replace('*', '').replace('&', 'and').replace('  ', ' '))
            diagnosis_str = ' '.join(diagnosis_text)
            hd5.create_dataset('ecg_rest_text', (1,), data=diagnosis_str, dtype=h5py.special_dtype(vlen=str))

    lead_data = record.p_signal
    for i, sig in enumerate(record.sig_name):
        dataset_name = 'strip_' + str(sig)
        hd5.create_dataset(rest_group + dataset_name, data=lead_data[i, :], compression='gzip', dtype=np.float)
        stats[dataset_name] += 1

    for i, sig in enumerate(record.sig_name):
        dataset_name = 'annotation_' + str(sig)
        ann = wfdb.rdann(xmlp, ann_ext + '_' + sig)
        hd5.create_dataset(annotation_group + dataset_name, data=ann.sample, compression='gzip', dtype=np.int)
