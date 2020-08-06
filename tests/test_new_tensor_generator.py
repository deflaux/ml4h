import pytest
import numpy as np

from ml4cvd.new_tensor_generator import tensor_generator_from_tensor_maps, _hd5_path_to_sample_id, sample_getter_from_tensor_maps
from ml4cvd.test_utils import TMAPS_UP_TO_4D
from ml4cvd.test_utils import build_hdf5s


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
    ).as_numpy_iterator()
    sample_id_to_path = {_hd5_path_to_sample_id(path): path for path in paths}
    for (inp, out), sample_id in zip(gen, sample_id_to_path.keys()):
        path = sample_id_to_path[sample_id]
        for tmap in TMAPS_UP_TO_4D:
            assert np.array_equal(expected_tensors[path, tmap], inp[tmap.input_name()])
            assert np.array_equal(expected_tensors[path, tmap], out[tmap.output_name()])


def test_sample_getter_from_tensor_maps(expected_tensors):
    paths = [path for path, _ in expected_tensors]
    sample_id_to_path = {_hd5_path_to_sample_id(path): path for path in paths}
    path_to_sample_id = {path: sample_id for sample_id, path in sample_id_to_path.items()}
    getter = sample_getter_from_tensor_maps(sample_id_to_path, TMAPS_UP_TO_4D, TMAPS_UP_TO_4D, False)
    for (path, tm), value in expected_tensors.items():
        fetched = getter(path_to_sample_id[path])
        assert (fetched[0][0][tm.input_name()] == value).all()
        assert (fetched[0][1][tm.output_name()] == value).all()
