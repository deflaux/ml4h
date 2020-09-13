import sys
import mock
import pytest

from ml4cvd.arguments import parse_args, TMAPS
from ml4cvd.test_utils import TMAPS as MOCK_TMAPS
from ml4cvd.test_utils import build_hdf5s


def pytest_configure():
    pytest.N_TENSORS = 50


@pytest.fixture(scope='class')
@mock.patch.dict(TMAPS, MOCK_TMAPS)
def default_arguments(tmpdir_factory):
    temp_dir = tmpdir_factory.mktemp('data')
    tensor_values = build_hdf5s(temp_dir, MOCK_TMAPS.values(), n=pytest.N_TENSORS)
    hdf5_dir = str(temp_dir)
    inp_key = '3d_cont'
    out_key = '1d_cat'
    sys.argv = [
        '',
        '--output_folder', hdf5_dir,
        '--input_tensors', inp_key,
        '--output_tensors', out_key,
        '--tensors', hdf5_dir,
        '--pool_x', '1',
        '--pool_y', '1',
        '--pool_z', '1',
        '--training_steps', '2',
        '--test_steps', '3',
        '--validation_steps', '2',
        '--epochs', '2',
        '--num_workers', '1',
        '--batch_size', '2',
    ]
    args = parse_args()
    args.tensor_values = tensor_values
    return args
