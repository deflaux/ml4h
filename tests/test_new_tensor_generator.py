import pytest
import numpy as np
import pandas as pd
import tensorflow as tf
from collections import defaultdict

from ml4cvd.new_tensor_generator import dataset_from_tensor_maps, _hd5_path_to_sample_id, sample_getter_from_tensor_maps
from ml4cvd.new_tensor_generator import dataset_from_sample_getter, DataFrameTensorGetter, SampleGetter, SampleIdStateSetter
from ml4cvd.new_tensor_generator import tensor_maps_to_output_shapes, tensor_maps_to_output_types, TensorMapTensorGetter
from ml4cvd.new_tensor_generator import HD5StateSetter
from ml4cvd.defines import SAMPLE_ID
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
    gen = dataset_from_tensor_maps(
        hd5_paths=list(paths),
        tensor_maps_in=TMAPS_UP_TO_4D, tensor_maps_out=TMAPS_UP_TO_4D,
    )
    gen = gen.as_numpy_iterator()
    sample_id_to_path = {_hd5_path_to_sample_id(path): path for path in paths}
    for (inp, out), sample_id in zip(gen, sorted(sample_id_to_path.keys())):
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


def test_model_trains(expected_tensors):
    paths = [path for path, _ in expected_tensors]
    gen = dataset_from_tensor_maps(
        hd5_paths=list(paths),
        tensor_maps_in=TMAPS_UP_TO_4D, tensor_maps_out=TMAPS_UP_TO_4D,
    ).batch(7)
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


def test_data_frame_tensor_getter():
    col = 'nice_col'
    df = pd.DataFrame({col: np.random.randn(pytest.N_TENSORS)})
    tensor_getter = DataFrameTensorGetter(df, col)
    sample_getter = SampleGetter([tensor_getter], [], [SampleIdStateSetter()])
    for i in range(pytest.N_TENSORS):
        assert (df.loc[i] == sample_getter(i)[0][0][col]).all()


def test_combine_tensor_maps_data_frame(expected_tensors):
    """
    Makes SampleGetter from dataframe and tmaps
    The dataframe's columns are the means of the tmaps
    """
    df = defaultdict(list)
    for (path, tm), value in expected_tensors.items():
        sample_id = _hd5_path_to_sample_id(path)
        if sample_id not in df[SAMPLE_ID]:
            df[SAMPLE_ID].append(sample_id)
        df[tm.input_name()].append(value.mean())
    df = pd.DataFrame(df)
    df.index = df[SAMPLE_ID]
    del df[SAMPLE_ID]

    hd5_paths = [path for path, _ in expected_tensors]
    sample_id_to_path = {_hd5_path_to_sample_id(path): path for path in hd5_paths}
    output_types = tensor_maps_to_output_types(TMAPS_UP_TO_4D, [])
    output_shapes = tensor_maps_to_output_shapes(TMAPS_UP_TO_4D, [])
    for col in df.columns:
        output_types[1][col] = tf.float32
        output_shapes[1][col] = tuple()

    sample_getter = SampleGetter(
        [TensorMapTensorGetter(tmap, is_input=True, augment=False) for tmap in TMAPS_UP_TO_4D],
        [DataFrameTensorGetter(df, col) for col in df.columns],
        [SampleIdStateSetter(), HD5StateSetter(sample_id_to_path)],
    )
    dataset = dataset_from_sample_getter(
        sample_getter, list(sample_id_to_path.keys()), output_types, output_shapes,
    )
    for inp, out in dataset.as_numpy_iterator():
        for name, val in inp.items():
            assert pytest.approx(out[name]) == val.mean()
