import os
import tempfile
from typing import Any, Callable, Type

import pandas as pd
import pytest
from fsspec.implementations.local import LocalFileSystem
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

from mlem.api import init, save
from mlem.core.dataset_type import Dataset, DatasetReader, DatasetWriter
from mlem.core.metadata import load_meta
from mlem.core.objects import DatasetMeta, ModelMeta

RESOURCES = "resources"

long = pytest.mark.long


def resource_path(test_file, *paths):
    resources_dir = os.path.join(os.path.dirname(test_file), RESOURCES)
    return os.path.join(resources_dir, *paths)


@pytest.fixture
def local_fs():
    return LocalFileSystem()


@pytest.fixture(params=["numpy", "pandas"])
def model_X_y(request):
    """Note that in tests we often use both model and data,
    so having them compatible is a requirement for now.
    Though in future we may want to add tests with incompatible models and data
    """
    if request.param == "numpy":
        X, y = load_iris(return_X_y=True)
    elif request.param == "pandas":
        X, y = load_iris(return_X_y=True)
        X = pd.DataFrame(X)
    model = DecisionTreeClassifier().fit(X, y)
    return model, X, y


@pytest.fixture
def pandas_data():
    return pd.DataFrame([[1, 0], [0, 1]], columns=["a", "b"])


@pytest.fixture
def X(model_X_y):
    model, X, y = model_X_y
    return X


@pytest.fixture
def model(model_X_y):
    model, X, y = model_X_y
    return model


@pytest.fixture
def dataset_meta(X):
    return DatasetMeta.from_data(X)


@pytest.fixture
def model_meta(model):
    return ModelMeta.from_obj(model)


@pytest.fixture
def model_path(model_X_y, tmpdir_factory):
    temp_dir = str(tmpdir_factory.mktemp("saved-model"))
    model, X, y = model_X_y
    # because of link=False we test reading by path here
    # reading by link name is not tested
    save(model, temp_dir, tmp_sample_data=X, link=False)
    yield temp_dir


@pytest.fixture
def data_path(X, tmpdir_factory):
    temp_dir = str(tmpdir_factory.mktemp("saved-data"))
    save(X, temp_dir, link=False)
    yield temp_dir


@pytest.fixture
def dataset_meta_saved(data_path):
    return load_meta(data_path)


@pytest.fixture
def model_meta_saved(model_path):
    return load_meta(model_path)


@pytest.fixture
def mlem_root(tmpdir_factory):
    dir = str(tmpdir_factory.mktemp("mlem-root"))
    init(dir)
    yield dir


@pytest.fixture
def model_path_mlem_root(model_X_y, tmpdir_factory):
    model, X, y = model_X_y
    dir = str(tmpdir_factory.mktemp("mlem-root-with-model"))
    init(dir)
    model_dir = os.path.join(dir, "generated-model")
    save(model, model_dir, tmp_sample_data=X, link=True)
    yield model_dir, dir


def dataset_write_read_check(
    dataset: Dataset,
    writer: DatasetWriter = None,
    reader_type: Type[DatasetReader] = None,
    custom_eq: Callable[[Any, Any], bool] = None,
    custom_assert: Callable[[Any, Any], Any] = None,
):
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = writer or dataset.dataset_type.get_writer()

        fs = LocalFileSystem()
        reader, artifacts = writer.write(dataset, fs, tmpdir)
        if reader_type is not None:
            assert isinstance(reader, reader_type)

        new = reader.read(fs, tmpdir)

        assert dataset.dataset_type == new.dataset_type
        if custom_assert is not None:
            custom_assert(new.data, dataset.data)
        else:
            if custom_eq is not None:
                assert custom_eq(new.data, dataset.data)
            else:
                assert new.data == dataset.data
