import pytest

from ml4cvd.new_tensor_generator import tensor_generator_from_tensor_maps
from ml4cvd.tensor_generators import _identity_batch, BATCH_INPUT_INDEX, BATCH_OUTPUT_INDEX
from ml4cvd.test_utils import TMAPS_UP_TO_4D
from ml4cvd.test_utils import build_hdf5s


def test_tensor_generator_from_tensor_maps(tmpdir_factory):
    temp_dir = tmpdir_factory.mktemp('explore_tensors')
    tmaps = TMAPS_UP_TO_4D
    expected = build_hdf5s(temp_dir, tmaps, n=pytest.N_TENSORS)
    paths = [path for path, _ in expected]
    gen = tensor_generator_from_tensor_maps(
        hd5_paths=list(paths),
        tensor_maps_in=TMAPS_UP_TO_4D, tensor_maps_out=TMAPS_UP_TO_4D,
        batch_size=1, batch_function=_identity_batch, return_ids=True,
    )
    sample_id_to_path = gen.state_setters[0].hd5_paths
    path_to_sample_id = {path: sample_id for sample_id, path in sample_id_to_path.items()}
    for (path, tm), value in expected.items():
        fetched = gen.get_tensors(path_to_sample_id[path])
        assert (fetched[BATCH_INPUT_INDEX][tm.name] == value).all()
        assert (fetched[BATCH_OUTPUT_INDEX][tm.name] == value).all()
