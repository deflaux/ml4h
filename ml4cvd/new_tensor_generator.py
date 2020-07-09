from typing import Set, Any, Dict, List, Callable, Tuple, Optional
from abc import ABC, abstractmethod
import numpy as np
import h5py

from ml4cvd.TensorMap import TensorMap


class StateSetter(ABC):

    @abstractmethod
    def get_state(self, sample_id: int) -> Any:
        pass

    @staticmethod
    @abstractmethod
    def get_name() -> str:
        pass


Batch = Dict[str, np.ndarray]
BatchFunction = Callable[[Batch, Batch, List[int]], Any]


class TensorGetter(ABC):

    required_states: Set[str]
    name: str
    shape: Tuple[int, ...]

    @abstractmethod
    def get_tensor(self, evaluated_states: Dict[str, Any]) -> np.ndarray:
        pass


# TODO: main thread batch, handle errors, different sample_id iterators
class TensorGenerator:

    def __init__(
            self, input_tensor_getters: List[TensorGetter], output_tensor_getters: List[TensorGetter],
            sample_ids: List[int],
            batch_size: int, batch_function: BatchFunction,
            state_setters: List[StateSetter],
            return_ids: bool = False,
            num_workers: int = 0,
    ):
        self.input_tensor_getters = input_tensor_getters
        self.output_tensor_getters = output_tensor_getters
        self.sample_ids = sample_ids
        self.batch_size = batch_size
        self.batch_function = batch_function
        self.return_ids = return_ids
        self.state_setters = state_setters
        self._check_states()
        self.num_workers = num_workers

    def _check_states(self):
        all_state_names = {state.get_name() for state in self.state_setters}
        for getter in self.input_tensor_getters + self.output_tensor_getters:
            missing_states = getter.required_states - all_state_names
            if missing_states:
                raise ValueError(
                    f'TensorGetter {getter.name} is missing required StateGetters {missing_states}.'
                )

    def _evaluate_states(self, sample_id: int):
        return {
            state.get_name(): state.get_state(sample_id)
            for state in self.state_setters
        }

    def get_tensors(self, sample_id: int) -> Tuple[Batch, Batch]:
        """Get one sample worth of tensors"""
        evaluated_states = self._evaluate_states(sample_id)
        sample_in = {}
        for tensor_getter in self.input_tensor_getters:
            tensor = tensor_getter.get_tensor(evaluated_states)
            sample_in[tensor_getter.name] = tensor
        sample_out = {}
        for tensor_getter in self.output_tensor_getters:
            tensor = tensor_getter.get_tensor(evaluated_states)
            sample_out[tensor_getter.name] = tensor
        return sample_in, sample_out

    def __next__(self) -> Tuple[Batch, Batch, Optional[List[int]]]:
        pass  # TODO multiprocess generate batch


class HD5State(StateSetter):

    def __init__(self, hd5_paths: Dict[int, str]):
        self.hd5_paths = hd5_paths

    # TODO: should be context_manager?
    def get_state(self, sample_id: int) -> h5py.File:
        return h5py.File(self.hd5_paths[sample_id], 'r')

    @staticmethod
    def get_name() -> str:
        return 'hd5_state'


class TensorMapTensorGetter(TensorGetter):

    def __init__(self, tensor_map: TensorMap, augment: bool):
        self.required_states = {HD5State.get_name()}
        self.required_state = HD5State.get_name()
        self.tensor_map = tensor_map
        self.name = tensor_map.name
        self.augment = augment

    def get_tensor(self, evaluated_states: Dict[str, Any]) -> np.ndarray:
        hd5 = evaluated_states[self.required_state]
        tensor = self.tensor_map.tensor_from_file(
            self.tensor_map, hd5,
        )
        return self.tensor_map.postprocess_tensor(tensor, self.augment, hd5)


def tensor_generator_from_tensor_maps(
        hd5_paths: List[str],
        tensor_maps_in: List[TensorMap], tensor_maps_out: List[TensorMap],
        batch_size: int, batch_function: BatchFunction,
        return_ids: bool = False,
        num_workers: int = 0,
        augment: bool = False,
) -> TensorGenerator:
    """For backwards compatibility."""
    sample_id_to_path = {i: path for i, path in enumerate(hd5_paths)}
    hd5_state = HD5State(sample_id_to_path)
    input_tensor_getters = [TensorMapTensorGetter(tmap, augment) for tmap in tensor_maps_in]
    output_tensor_getters = [TensorMapTensorGetter(tmap, augment) for tmap in tensor_maps_out]
    return TensorGenerator(
        input_tensor_getters=input_tensor_getters, output_tensor_getters=output_tensor_getters,
        sample_ids=list(sample_id_to_path), state_setters=[hd5_state],
        batch_size=batch_size, batch_function=batch_function, return_ids=return_ids,
        num_workers=num_workers,
    )
