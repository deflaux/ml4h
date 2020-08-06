import os
from typing import Set, Any, Dict, List, Callable, Tuple, Optional
from contextlib import contextmanager, ExitStack
from abc import ABC, abstractmethod
import numpy as np
import h5py
import tensorflow as tf

from ml4cvd.TensorMap import TensorMap, Interpretation


class StateSetter(ABC):

    @abstractmethod
    def get_state(self, sample_id: int) -> Any:
        pass

    @staticmethod
    @abstractmethod
    def get_name() -> str:
        pass


class TensorGetter(ABC):

    required_states: Set[str]
    name: str
    shape: Tuple[int, ...]

    @abstractmethod
    @contextmanager
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
        all_state_names = {state.get_name() for state in self.state_setters}
        for getter in self.input_tensor_getters + self.output_tensor_getters:
            missing_states = getter.required_states - all_state_names
            if missing_states:
                raise ValueError(
                    f'TensorGetter {getter.name} is missing required StateGetters {missing_states}.'
                )

    @contextmanager
    def _evaluate_states(self, sample_id: int) -> Dict[str, Any]:
        with ExitStack() as stack:
            yield {
                state.get_name(): stack.enter_context(state.get_state(sample_id))
                for state in self.state_setters
            }

    def __call__(self, sample_id: int) -> List[Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]]:
        with self._evaluate_states(sample_id) as evaluated_states:
            return [(
                {tensor_getter.name: tensor_getter.get_tensor(evaluated_states) for tensor_getter in self.input_tensor_getters},
                {tensor_getter.name: tensor_getter.get_tensor(evaluated_states) for tensor_getter in self.output_tensor_getters},
            )]


class HD5State(StateSetter):

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
        self.required_states = {HD5State.get_name()}
        self.required_state = HD5State.get_name()
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


INTERPRETATION_TO_TENSOR_FLOW_TYPE = {  # TODO: what types?
    Interpretation.CONTINUOUS: tf.float32,
    Interpretation.CATEGORICAL: tf.int16,
    Interpretation.EMBEDDING: tf.float32,
    Interpretation.LANGUAGE: tf.float32,
    Interpretation.TIME_TO_EVENT: tf.float64,
    Interpretation.SURVIVAL_CURVE: tf.float64,
    Interpretation.DISCRETIZED: tf.int32,
    Interpretation.MESH: tf.float64,
}


def tensor_maps_to_output_types(tensor_maps_in: List[TensorMap], tensor_maps_out: List[TensorMap]):
    return (
        {tmap.input_name(): INTERPRETATION_TO_TENSOR_FLOW_TYPE[tmap.interpretation] for tmap in tensor_maps_in},
        {tmap.output_name(): INTERPRETATION_TO_TENSOR_FLOW_TYPE[tmap.interpretation] for tmap in tensor_maps_out},
    )


def tensor_maps_to_output_shapes(tensor_maps_in: List[TensorMap], tensor_maps_out: List[TensorMap]):
    return (
        {tmap.input_name(): tmap.shape for tmap in tensor_maps_in},
        {tmap.output_name(): tmap.shape for tmap in tensor_maps_out},
    )


def sample_getter_from_tensor_maps(
        sample_id_to_path: Dict[int, str],
        tensor_maps_in: List[TensorMap], tensor_maps_out: List[TensorMap],
        augment: bool = False,
) -> SampleGetter:
    hd5_state = HD5State(sample_id_to_path)
    input_tensor_getters = [TensorMapTensorGetter(tmap, augment, True) for tmap in tensor_maps_in]
    output_tensor_getters = [TensorMapTensorGetter(tmap, augment, False) for tmap in tensor_maps_out]
    return SampleGetter(
        input_tensor_getters=input_tensor_getters, output_tensor_getters=output_tensor_getters,
        state_setters=[hd5_state],
    )


def tensor_generator_from_tensor_maps(
        hd5_paths: List[str],
        tensor_maps_in: List[TensorMap], tensor_maps_out: List[TensorMap],
        batch_size: int,
        augment: bool = False,  # TODO: number of workers, deterministism, shuffling
) -> tf.data.Dataset:
    sample_id_to_path = {_hd5_path_to_sample_id(path): path for path in hd5_paths}
    sample_getter = sample_getter_from_tensor_maps(sample_id_to_path, tensor_maps_in, tensor_maps_out, augment)
    output_types = tensor_maps_to_output_types(tensor_maps_in, tensor_maps_out)
    output_shapes = tensor_maps_to_output_shapes(tensor_maps_in, tensor_maps_out)
    return tf.data.Dataset.from_tensor_slices(  # TODO: This feels overly complicated
        list(sample_id_to_path.keys())).interleave(  # TODO: should the sample_id dataset be passed as an argument?
        lambda sample_id: tf.data.Dataset.from_generator(
            sample_getter, args=(sample_id,),
            output_types=output_types,
            output_shapes=output_shapes,
        )
    ).batch(batch_size)
