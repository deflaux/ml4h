import pytest
import numpy as np

from ml4cvd.new_tensor_generator import tensor_generator_from_tensor_maps, _hd5_path_to_sample_id, sample_getter_from_tensor_maps
from ml4cvd.test_utils import TMAPS_UP_TO_4D
from ml4cvd.test_utils import build_hdf5s
from ml4cvd.models import make_multimodal_multitask_model, BottleneckType


@pytest.fixture(scope='session')
def expected_tensors(tmpdir_factory):
    temp_dir = tmpdir_factory.mktemp('explore_tensors')
    tmaps = TMAPS_UP_TO_4D
    return build_hdf5s(temp_dir, tmaps, n=pytest.N_TENSORS)


def test_tensor_generator_from_tensor_maps(expected_tensors):
    paths = [path for path, _ in expected_tensors]
    gen = tensor_generator_from_tensor_maps(
        hd5_paths=list(paths),
        tensor_maps_in=TMAPS_UP_TO_4D, tensor_maps_out=TMAPS_UP_TO_4D,
        batch_size=1,
    )
    gen = gen.as_numpy_iterator()
    sample_id_to_path = {_hd5_path_to_sample_id(path): path for path in paths}
    for (inp, out), sample_id in zip(gen, sample_id_to_path.keys()):
        path = sample_id_to_path[sample_id]
        for tmap in TMAPS_UP_TO_4D:
            assert np.array_equal(expected_tensors[path, tmap], inp[tmap.input_name()][0])  # 0 since batch_size is 1
            assert np.array_equal(expected_tensors[path, tmap], out[tmap.output_name()][0])


def test_sample_getter_from_tensor_maps(expected_tensors):
    paths = [path for path, _ in expected_tensors]
    sample_id_to_path = {_hd5_path_to_sample_id(path): path for path in paths}
    path_to_sample_id = {path: sample_id for sample_id, path in sample_id_to_path.items()}
    getter = sample_getter_from_tensor_maps(sample_id_to_path, TMAPS_UP_TO_4D, TMAPS_UP_TO_4D, False)
    for (path, tm), value in expected_tensors.items():
        fetched = getter(path_to_sample_id[path])
        assert (fetched[0][0][tm.input_name()] == value).all()
        assert (fetched[0][1][tm.output_name()] == value).all()


def test_model_trains(expected_tensors):
    paths = [path for path, _ in expected_tensors]
    gen = tensor_generator_from_tensor_maps(
        hd5_paths=list(paths),
        tensor_maps_in=TMAPS_UP_TO_4D, tensor_maps_out=TMAPS_UP_TO_4D,
        batch_size=7,
    )
    model_params = {  # TODO: this shouldn't be here
        'activation': 'relu',
        'dense_layers': [4, 2],
        'dense_blocks': [5, 3],
        'block_size': 3,
        'conv_width': 3,
        'learning_rate': 1e-3,
        'optimizer': 'adam',
        'conv_type': 'conv',
        'conv_layers': [6, 5, 3],
        'conv_x': [3],
        'conv_y': [3],
        'conv_z': [2],
        'padding': 'same',
        'max_pools': [],
        'pool_type': 'max',
        'pool_x': 1,
        'pool_y': 1,
        'pool_z': 1,
        'conv_regularize': 'spatial_dropout',
        'conv_regularize_rate': .1,
        'conv_normalize': 'batch_norm',
        'dense_regularize': 'dropout',
        'dense_regularize_rate': .1,
        'dense_normalize': 'batch_norm',
        'bottleneck_type': BottleneckType.FlattenRestructure,
    }
    m = make_multimodal_multitask_model(
        TMAPS_UP_TO_4D,
        TMAPS_UP_TO_4D,
        **model_params,
    )
    m.fit(gen, epochs=10)
