import os
import ray
import pandas as pd
from functools import partial
from typing import Set, Any, Dict, List, Tuple, Callable, Union
from contextlib import contextmanager, ExitStack
from abc import ABC, abstractmethod
import numpy as np
import h5py
import tensorflow as tf
from multiprocessing import Pool

from ml4cvd.TensorMap import TensorMap, Interpretation
from ml4cvd.defines import SAMPLE_ID

if not ray.is_initialized():
    ray.init()


Batch = Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]
DataSet = Union[ray.util.iter.ParallelIterator[Batch], ray.util.iter.LocalIterator[Batch]]


class StateSetter(ABC):

    @abstractmethod
    def get_state(self, sample_id: int) -> Any:
        pass

    @staticmethod
    @abstractmethod
    def get_name() -> str:
        pass


class SampleIdStateSetter(StateSetter):
    @staticmethod
    def get_name() -> str:
        return SAMPLE_ID

    @contextmanager
    def get_state(self, sample_id: int) -> int:
        yield sample_id


class TensorGetter(ABC):

    required_states: Set[str]
    name: str  # TODO: should this be an abstract property?
    shape: Tuple[int, ...]

    @abstractmethod
    def get_tensor(self, evaluated_states: Dict[str, Any]) -> np.ndarray:
        pass


class SampleGetter:
    """Gets all of the data for one sample_id"""
    def __init__(
            self,
            input_tensor_getters: List[TensorGetter],
            output_tensor_getters: List[TensorGetter],
            state_setters: List[StateSetter],
    ):
        self.input_tensor_getters = input_tensor_getters
        self.output_tensor_getters = output_tensor_getters
        self.state_setters = state_setters
        self._check_states()

    def _check_states(self):
        state_setters = {setter.get_name(): setter for setter in self.state_setters}
        all_state_names = set(state_setters)
        for getter in self.input_tensor_getters + self.output_tensor_getters:
            missing_states = getter.required_states - all_state_names
            if missing_states:
                raise ValueError(
                    f'TensorGetter {getter.name} is missing required StateGetters {missing_states}.'
                )
        self.state_setters = list(state_setters.values())

    @contextmanager
    def _evaluate_states(self, sample_id: int) -> Dict[str, Any]:
        with ExitStack() as stack:
            yield {
                state.get_name(): stack.enter_context(state.get_state(sample_id))
                for state in self.state_setters
            }

    def __call__(self, sample_id: int) -> Batch:
        with self._evaluate_states(sample_id) as evaluated_states:
            return (
                {tensor_getter.name: tensor_getter.get_tensor(evaluated_states) for tensor_getter in self.input_tensor_getters},
                {tensor_getter.name: tensor_getter.get_tensor(evaluated_states) for tensor_getter in self.output_tensor_getters},
            )

    def __add__(self, other: "SampleGetter"):
        return SampleGetter(
            self.input_tensor_getters + other.input_tensor_getters,
            self.output_tensor_getters + other.output_tensor_getters,
            self.state_setters + other.state_setters,
        )


class HD5StateSetter(StateSetter):

    def __init__(self, hd5_paths: Dict[int, str]):
        self.hd5_paths = hd5_paths

    @contextmanager
    def get_state(self, sample_id: int) -> h5py.File:
        with h5py.File(self.hd5_paths[sample_id], 'r') as hd5:
            yield hd5

    @staticmethod
    def get_name() -> str:
        return 'hd5_state'


class TensorMapTensorGetter(TensorGetter):

    def __init__(self, tensor_map: TensorMap, augment: bool, is_input: bool):
        self.required_states = {HD5StateSetter.get_name()}
        self.required_state = HD5StateSetter.get_name()
        self.tensor_map = tensor_map
        self.name = tensor_map.input_name() if is_input else tensor_map.output_name()
        self.augment = augment

    def get_tensor(self, evaluated_states: Dict[str, Any]) -> np.ndarray:
        hd5 = evaluated_states[self.required_state]
        tensor = self.tensor_map.tensor_from_file(
            self.tensor_map, hd5,
        )
        return self.tensor_map.postprocess_tensor(tensor, self.augment, hd5)


def _hd5_path_to_sample_id(path: str) -> int:
    return int(os.path.basename(path).replace('.hd5', ''))


def dataset_from_sample_getter(
        sample_getter: SampleGetter,
        sample_ids: List[int],
        num_workers: int,
) -> DataSet:
    return ray.util.iter.from_items(sample_ids, num_shards=num_workers).for_each(sample_getter)


def _merge_batch_in_or_out(x: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    keys = x[0].keys()
    return {key: np.stack([x[key] for x in x]) for key in keys}


def merge_batch(batch: List[Batch]) -> Batch:
    return _merge_batch_in_or_out([x[0] for x in batch]), _merge_batch_in_or_out([x[1] for x in batch])


def batch_dataset(dataset: DataSet, batch_size: int) -> DataSet:
    return dataset.batch(batch_size).for_each(merge_batch)


def sample_getter_from_tensor_maps(
        sample_id_to_path: Dict[int, str],
        tensor_maps_in: List[TensorMap], tensor_maps_out: List[TensorMap],
        augment: bool = False,
) -> SampleGetter:
    hd5_state = HD5StateSetter(sample_id_to_path)
    input_tensor_getters = [TensorMapTensorGetter(tmap, augment, True) for tmap in tensor_maps_in]
    output_tensor_getters = [TensorMapTensorGetter(tmap, augment, False) for tmap in tensor_maps_out]
    return SampleGetter(
        input_tensor_getters=input_tensor_getters, output_tensor_getters=output_tensor_getters,
        state_setters=[hd5_state],
    )


def dataset_from_tensor_maps(
        hd5_paths: List[str],
        tensor_maps_in: List[TensorMap], tensor_maps_out: List[TensorMap],
        augment: bool,
        epochs: int,
        num_workers: int,
        batch_size: int,
        prepare_sample_ids: Callable[[List[int], int], List[int]],
) -> DataSet:
    sample_id_to_path = {_hd5_path_to_sample_id(path): path for path in hd5_paths}
    sample_getter = sample_getter_from_tensor_maps(
        sample_id_to_path, tensor_maps_in, tensor_maps_out, augment
    )
    sample_ids = prepare_sample_ids(list(sample_id_to_path.keys()), epochs)
    print(f'sample ids {len(sample_ids)}')
    dataset = dataset_from_sample_getter(
        sample_getter, sample_ids, num_workers,
    )
    return batch_dataset(dataset, batch_size)


def prepare_train_sample_ids(sample_ids: List[int], epochs: int) -> List[int]:
    """Concatenates a list of sample ids shuffled every epoch"""
    return sum((np.random.permutation(sample_ids).tolist() for _ in range(epochs)), [])


def train_dataset_from_tensor_maps(
        hd5_paths: List[str],
        tensor_maps_in: List[TensorMap], tensor_maps_out: List[TensorMap],
        epochs: int,
        num_workers: int,
        batch_size: int,
) -> DataSet:
    return dataset_from_tensor_maps(
        hd5_paths,
        tensor_maps_in, tensor_maps_out,
        True,
        epochs,
        num_workers,
        batch_size,
        prepare_train_sample_ids,
    ).gather_async()


def valid_dataset_from_tensor_maps(
        hd5_paths: List[str],
        tensor_maps_in: List[TensorMap], tensor_maps_out: List[TensorMap],
        epochs: int,
        num_workers: int,
        batch_size: int,
) -> DataSet:
    return dataset_from_tensor_maps(
        hd5_paths,
        tensor_maps_in, tensor_maps_out,
        False,
        epochs,
        num_workers,
        batch_size,
        prepare_train_sample_ids,
    ).gather_async()


def test_dataset_from_tensor_maps(
        hd5_paths: List[str],
        tensor_maps_in: List[TensorMap], tensor_maps_out: List[TensorMap],
        num_workers: int,
        batch_size: int,
) -> DataSet:
    return dataset_from_tensor_maps(
        hd5_paths,
        tensor_maps_in, tensor_maps_out,
        False,
        1,
        num_workers,
        batch_size,
        lambda ids, _: ids,
    ).gather_sync()


class DataFrameTensorGetter(TensorGetter):
    def __init__(self, df: pd.DataFrame, column: str):
        """df should be indexed by sample_id"""
        self.name = column
        self.required_state = SampleIdStateSetter.get_name()
        self.required_states = {self.required_state}
        self.df = df[column]

    def get_tensor(self, evaluated_states: Dict[str, Any]) -> np.ndarray:
        return self.df[evaluated_states[self.required_state]]


ERROR_COL = 'error'


def _format_error(error: Exception):
    return f'{type(error).__name__}: {error}'


def try_sample_id(sample_id: int, sample_getter: SampleGetter) -> Tuple[float, str]:
    # TODO: this could be part of SampleGetter, and keep track of the getter the error came from
    try:
        sample_getter(sample_id)
    except (IndexError, KeyError, ValueError, OSError, RuntimeError) as error:
        return sample_id, _format_error(error)
    return sample_id, ''


def find_working_ids(
        sample_getter: SampleGetter, sample_ids: List[int], num_workers: int,
) -> pd.DataFrame:
    # TODO: this should also use ray
    pool = Pool(num_workers)
    tried_sample_ids = []
    errors = []
    for i, (sample_id, error) in enumerate(
            pool.imap_unordered(partial(try_sample_id, sample_getter=sample_getter), sample_ids)
    ):
        tried_sample_ids.append(sample_id)
        errors.append(error)
        print(f'{(i + 1) / len(sample_ids):.2%} done finding working sample ids', end='\r')
    return pd.DataFrame({SAMPLE_ID: tried_sample_ids, ERROR_COL: errors})
