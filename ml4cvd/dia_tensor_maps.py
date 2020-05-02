import time
import h5py
from typing import Tuple, Callable, List, Dict, Optional, Iterator
import numpy as np
from collections import Counter
from multiprocessing import Pool
import pandas as pd
from functools import partial


"""
Hierarchy of hdf5s is
/file_type/instance/name/value
e.g. DIA/instance_10/Date_Age/23.4
"""
AGE = 'DateAge'
DIA = 'Dia'
CODE = 'Code'
CODE_TYPE = 'Code_Type'
INPATIENT_OUTPATIENT = 'Inpatient_Outpatient'
DIAGNOSIS_FLAG = 'Diagnosis_Flag'
DIAGNOSIS_NAME = 'Diagnosis_Name'
KNOWN_NAMES = [
    'Clinic', 'Code', 'Code_Type', 'Date_Age', 'Diagnosis_Flag',
    'Diagnosis_Name', 'Encounter_number', 'Hospital',
    'Inpatient_Outpatient', 'Provider', 'file_name',
]
KNOWN_CODE_TYPES = [
    'ICD9', 'LMR', 'DRG', 'Phenotype', 'APDRG', 'DSM4', 'Oncall', 'ICD10', 'APRDRG',
]
KNOWN_INPATIENT_OUTPATIENT = [
    'Outpatient', 'Inpatient', 'not recorded', 'Unknown', 'Outpatient-Emergency',
]
ICD9 = 'ICD9'
ICD10 = 'ICD10'
HF_ICD9 = {
    '398.91',
    '402.01',
    '402.11',
    '402.91',
    '404.01',
    '404.03',
    '404.11',
    '404.13',
    '404.91',
    '404.93',
    '428.0',
    '428.1',
    '428.2',
    '428.21',
    '428.22',
    '428.3',
    '428.31',
    '428.32',
    '428.33',
    '428.4',
    '428.41',
    '428.42',
    '428.43',
    '428.9',
}
HF_ICD10 = {
    'I11.0',
    'I13.0',
    'I13.2',
    'I132.25',
    'I50.0',
    'I50.1',
    'I150.20',
    'I50.21',
    'I50.22',
    'I50.23',
    'I50.30',
    'I50.31',
    'I50.32',
    'I50.33',
    'I50.40',
    'I50.41',
    'I50.42',
    'I50.43',
    'I50.9',
}


# Instance filtering
def _all_instances(hd5: h5py.File, file_type: str):
    for instance in hd5[file_type]:
        yield hd5[file_type][instance]


def _instances_with_flag(hd5: h5py.File, file_type: str, name: str) -> h5py.Group:
    for instance in hd5[file_type]:
        if name in instance:
            yield instance


def _filter_instances(
        hd5: h5py.File, file_type: str, instance_filter: Callable[[h5py.Group], bool],
) -> h5py.Group:
    yield from filter(instance_filter, _all_instances(hd5, file_type))


def _instances_filtered_by_value(
        hd5: h5py.File, file_type: str, name: str, value_filter: Callable[[h5py.Group], bool],
) -> h5py.Group:
    yield from filter(value_filter, _instances_with_flag(hd5, file_type, name))


def _and_instance_filters(a: Callable[[h5py.Group], bool], b: Callable[[h5py.Group], bool]) -> Callable[[h5py.Group], bool]:
    return lambda instance: a(instance) and b(instance)


def _no_filter(instance: h5py.Group) -> bool:
    return True


# Generic use
def _str_from_instance(instance: h5py.Group, name: str) -> str:
    return instance[name][()]


def _optional_str_from_instance(instance: h5py.Group, name: str) -> Optional[str]:
    if name in instance:
        return instance[name][()]


def _strs_from_instance(instance: h5py.Group, names: List[str]) -> Dict[str, str]:
    """Assumes names all exist"""
    return {name: _str_from_instance(instance, name) for name in names}


def _optional_strs_from_instance(instance: h5py.Group, names: List[str]) -> Dict[str, str]:
    """Assumes names all exist"""
    return {name: _optional_str_from_instance(instance, name) for name in names}


def _optional_strs_from_instance_tuple(instance: h5py.Group, names: List[str]) -> Tuple[str, ...]:
    """Assumes names all exist"""
    return tuple(_optional_str_from_instance(instance, name) for name in names)


# Age filtering
class AgeRange:
    def __init__(self, lower: float, upper: float):
        self.lower, self.upper = lower, upper

    def __contains__(self, item: float) -> bool:
        return self.lower <= item <= self.upper


def _age_from_instance(instance: h5py.Group) -> float:
    return instance[AGE][0]


def _age_filter(instance: h5py.Group, age_range: AgeRange) -> bool:
    if AGE in instance:
        return _age_from_instance(instance) in age_range
    return False


def _build_age_filter(age_range: AgeRange):
    return lambda instance: _age_filter(instance, age_range)


def _instances_with_flag_in_age_range(hd5: h5py.File, file_type: str, age_range: AgeRange) -> h5py.Group:
    """TODO: If instances are time ordered, then could be log n instead of n"""
    yield from _instances_filtered_by_value(hd5, file_type, AGE, _build_age_filter(age_range))


# ICD filtering
def _hf_code_filter(instance: h5py.Group) -> bool:
    if CODE in instance and CODE_TYPE in instance:
        code_type = _str_from_instance(instance, CODE_TYPE)
        if code_type == ICD9:
            return _str_from_instance(instance, CODE) in HF_ICD9
        if code_type == ICD10:
            return _str_from_instance(instance, CODE) in HF_ICD10
    return False


# EHR definition cohorts
def _age_below_20(hd5: h5py.File) -> h5py.Group:
    yield from _instances_with_flag_in_age_range(hd5, DIA, AgeRange(-np.inf, 20))


def cross_tab_names(instances: Iterator[h5py.Group], names: List[str]) -> Counter:
    counts = Counter()
    for instance in instances:
        counts[_optional_strs_from_instance_tuple(instance, names)] += 1
    return counts


def cross_tab_names_from_path(
        hd5_path: str, file_type: str, names: List[str], instance_filter: Callable[[h5py.Group], bool],
) -> Counter:
    counts = Counter()
    with h5py.File(hd5_path, 'r') as hd5:
        instances = _filter_instances(hd5, file_type, instance_filter)
        for instance in instances:
            counts[_optional_strs_from_instance_tuple(instance, names)] += 1
    return counts


def cross_tab_names_to_df(names: List[str], counter: Counter):
    rows = []
    for values, count in counter.items():
        row = {name: value for name, value in zip(names, values)}
        row['count'] = count
        rows.append(row)
    return pd.DataFrame(rows)


def cross_tab_multiprocess(
        file_type: str, names: List[str], instance_filter: Callable[[h5py.Group], bool],
        hd5_paths: List[str], num_workers: int,
) -> pd.DataFrame:
    hd5_func = partial(
        cross_tab_names_from_path,
        file_type=file_type, names=names, instance_filter=instance_filter,
    )
    pool = Pool(num_workers)
    print(f'Beginning cross tab of {names}.')
    now = time.time()
    counts = sum(pool.map(hd5_func, hd5_paths), Counter())
    dur = time.time() - now
    print(f'Cross tab took {dur:.2f} seconds at {len(hd5_paths) / dur:.3f} paths/second and {dur * num_workers / len(hd5_paths):.3f} worker seconds/path.')
    return cross_tab_names_to_df(names, counts)


def cross_tab_hf(hd5_paths: List[str]) -> pd.DataFrame:
    names = [CODE, CODE_TYPE, INPATIENT_OUTPATIENT, DIAGNOSIS_FLAG, DIAGNOSIS_NAME]
    return cross_tab_multiprocess(DIA, names, _hf_code_filter, hd5_paths, 4)


def cross_tab_inpatient_diagnosis_flag(hd5_paths: List[str]) -> pd.DataFrame:
    names = [INPATIENT_OUTPATIENT, DIAGNOSIS_FLAG]
    return cross_tab_multiprocess(DIA, names, _no_filter, hd5_paths, 4)


if __name__ == '__main__':
    import os
    d = '/data/cvrepo/loyalty-cohort-tensors/2020-04-24/'
    paths = [os.path.join(d, f) for f in os.listdir(d)]
    ct = cross_tab_hf(paths)
    ct.to_csv('hf_code_cross_tab.csv', index=False)
    ct = cross_tab_inpatient_diagnosis_flag(paths)
    ct.to_csv('inpatient_diagnosis_cross_tab.csv', index=False)
