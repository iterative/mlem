import os
import tempfile
from typing import Any, Callable, Type

import pandas as pd
import pytest
from fsspec.implementations.local import LocalFileSystem
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

from mlem import CONFIG
from mlem.api import init, save
from mlem.constants import PREDICT_ARG_NAME, PREDICT_METHOD_NAME
from mlem.contrib.sklearn import SklearnModel
from mlem.core.dataset_type import (
    Dataset,
    DatasetReader,
    DatasetType,
    DatasetWriter,
)
from mlem.core.metadata import load_meta
from mlem.core.model import Argument, ModelType, Signature
from mlem.core.objects import DatasetMeta, ModelMeta, mlem_dir_path
from mlem.core.requirements import Requirements

RESOURCES = "resources"

long = pytest.mark.long


@pytest.fixture(scope="session", autouse=True)
def add_test_env():
    os.environ["MLEM_TESTS"] = "true"
    CONFIG.TESTS = True


def resource_path(test_file, *paths):
    resources_dir = os.path.join(os.path.dirname(test_file), RESOURCES)
    return os.path.join(resources_dir, *paths)


@pytest.fixture
def local_fs():
    return LocalFileSystem()


@pytest.fixture(params=["numpy", "pandas"])
def model_train_target(request):
    """Note that in tests we often use both model and data,
    so having them compatible is a requirement for now.
    Though in future we may want to add tests with incompatible models and data
    """
    if request.param == "numpy":
        train, target = load_iris(return_X_y=True)
    elif request.param == "pandas":
        train, target = load_iris(return_X_y=True)
        train = pd.DataFrame(train)
    model = DecisionTreeClassifier().fit(train, target)
    return model, train, target


@pytest.fixture
def pandas_data():
    return pd.DataFrame([[1, 0], [0, 1]], columns=["a", "b"])


@pytest.fixture
def train(model_train_target):
    return model_train_target[1]


@pytest.fixture
def model(model_train_target):
    return model_train_target[0]


@pytest.fixture
def dataset_meta(train):
    return DatasetMeta.from_data(train)


@pytest.fixture
def model_meta(model):
    return ModelMeta.from_obj(model)


@pytest.fixture
def model_path(model_train_target, tmpdir_factory):
    temp_dir = str(tmpdir_factory.mktemp("saved-model"))
    model, train, _ = model_train_target
    # because of link=False we test reading by path here
    # reading by link name is not tested
    save(model, temp_dir, tmp_sample_data=train, link=False)
    yield temp_dir


@pytest.fixture
def data_path(train, tmpdir_factory):
    temp_dir = str(tmpdir_factory.mktemp("saved-data"))
    save(train, temp_dir, link=False)
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
    # TODO: bug: when reqs are empty, they serialize to "{}", not "[]"
    model = ModelMeta(
        requirements=Requirements.new("sklearn"),
        model_type=SklearnModel(methods={}, model=""),
    )
    model.dump("model1", mlem_root=dir)

    model.make_link(
        mlem_dir_path(
            "latest", LocalFileSystem(), obj_type=ModelMeta, mlem_root=dir
        )
    )
    yield dir


@pytest.fixture
def model_path_mlem_root(model_train_target, tmpdir_factory):
    model, train, _ = model_train_target
    dir = str(tmpdir_factory.mktemp("mlem-root-with-model"))
    init(dir)
    model_dir = os.path.join(dir, "generated-model")
    save(model, model_dir, tmp_sample_data=train, link=True)
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
        reader, _ = writer.write(dataset, fs, tmpdir)
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


def check_model_type_common_interface(
    model_type: ModelType, data_type: DatasetType, returns_type: DatasetType
):
    assert PREDICT_METHOD_NAME in model_type.methods
    assert model_type.methods[PREDICT_METHOD_NAME] == Signature(
        name=PREDICT_METHOD_NAME,
        args=[Argument(name=PREDICT_ARG_NAME, type_=data_type)],
        returns=returns_type,
    )
