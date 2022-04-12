import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from mlem.contrib.lightgbm import LightGBMDatasetType, LightGBMModel
from mlem.contrib.numpy import NumpyNdarrayType
from mlem.contrib.pandas import DataFrameType
from mlem.core.artifacts import LOCAL_STORAGE
from mlem.core.dataset_type import DatasetAnalyzer, DatasetType
from mlem.core.errors import DeserializationError, SerializationError
from mlem.core.model import ModelAnalyzer, ModelType
from mlem.core.requirements import UnixPackageRequirement
from tests.conftest import check_model_type_common_interface, long


@pytest.fixture
def np_payload():
    return np.linspace(0, 2, 5).reshape((-1, 1))


@pytest.fixture
def dataset_np(np_payload):
    return lgb.Dataset(
        np_payload, label=np_payload.reshape((-1,)), free_raw_data=False
    )


@pytest.fixture
def df_payload():
    return pd.DataFrame([{"a": i} for i in range(2)])


@pytest.fixture
def dataset_df(df_payload):
    return lgb.Dataset(
        df_payload,
        label=np.linspace(0, 2).reshape((-1, 1)),
        free_raw_data=False,
    )


@pytest.fixture
def booster(dataset_np):
    return lgb.train({}, dataset_np, 1)


@pytest.fixture
def model(booster, dataset_np) -> ModelType:
    return ModelAnalyzer.analyze(booster, sample_data=dataset_np)


@pytest.fixture
def dtype_np(dataset_np):
    return DatasetAnalyzer.analyze(dataset_np)


@pytest.fixture
def dtype_df(dataset_df):
    return DatasetAnalyzer.analyze(dataset_df)


def test_hook_np(dtype_np: DatasetType):
    assert set(dtype_np.get_requirements().modules) == {"lightgbm", "numpy"}
    assert isinstance(dtype_np, LightGBMDatasetType)
    assert isinstance(dtype_np.inner, NumpyNdarrayType)
    assert dtype_np.get_model().__name__ == dtype_np.inner.get_model().__name__
    assert dtype_np.get_model().schema() == {
        "title": "NumpyNdarray",
        "type": "array",
        "items": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 1,
            "maxItems": 1,
        },
    }


def test_hook_df(dtype_df: DatasetType):
    assert set(dtype_df.get_requirements().modules) == {"lightgbm", "pandas"}
    assert isinstance(dtype_df, LightGBMDatasetType)
    assert isinstance(dtype_df.inner, DataFrameType)
    assert dtype_df.get_model().__name__ == dtype_df.inner.get_model().__name__
    assert dtype_df.get_model().schema() == {
        "title": "DataFrame",
        "type": "object",
        "properties": {
            "values": {
                "title": "Values",
                "type": "array",
                "items": {"$ref": "#/definitions/DataFrameRow"},
            }
        },
        "required": ["values"],
        "definitions": {
            "DataFrameRow": {
                "title": "DataFrameRow",
                "type": "object",
                "properties": {"a": {"title": "A", "type": "integer"}},
                "required": ["a"],
            }
        },
    }


def test_serialize__np(dtype_np, np_payload):
    ds = lgb.Dataset(np_payload)
    payload = dtype_np.serialize(ds)
    assert payload == np_payload.tolist()

    with pytest.raises(SerializationError):
        dtype_np.serialize({"abc": 123})  # wrong type


def test_deserialize__np(dtype_np, np_payload):
    ds = dtype_np.deserialize(np_payload)
    assert isinstance(ds, lgb.Dataset)
    assert np.all(ds.data == np_payload)

    with pytest.raises(DeserializationError):
        dtype_np.deserialize([[1], ["abc"]])  # illegal matrix


def test_serialize__df(dtype_df, df_payload):
    ds = lgb.Dataset(df_payload)
    payload = dtype_df.serialize(ds)
    assert payload["values"] == df_payload.to_dict("records")


def test_deserialize__df(dtype_df, df_payload):
    ds = dtype_df.deserialize({"values": df_payload})
    assert isinstance(ds, lgb.Dataset)
    assert ds.data.equals(df_payload)


def test_hook(model, booster, dataset_np):
    assert isinstance(model, LightGBMModel)
    assert model.model == booster
    assert "lightgbm_predict" in model.methods
    data_type = DatasetAnalyzer.analyze(dataset_np)

    check_model_type_common_interface(
        model, data_type, NumpyNdarrayType(shape=(None,), dtype="float64")
    )


def test_model__predict(model, dataset_np):
    predict = model.call_method("predict", dataset_np)
    assert isinstance(predict, np.ndarray)
    assert len(predict) == dataset_np.num_data()


def test_model__predict_not_dataset(model):
    data = [[1]]
    predict = model.call_method("predict", data)
    assert isinstance(predict, np.ndarray)
    assert len(predict) == len(data)


@long
def test_model__dump_load(tmpdir, model, dataset_np, local_fs):
    # pandas is not required, but if it is installed, it is imported by lightgbm
    expected_requirements = {"lightgbm", "numpy", "scipy", "pandas"}
    assert set(model.get_requirements().modules) == expected_requirements

    artifacts = model.dump(LOCAL_STORAGE, tmpdir)

    model.unbind()
    with pytest.raises(ValueError):
        model.call_method("predict", dataset_np)

    model.load(artifacts)
    test_model__predict(model, dataset_np)

    assert set(model.get_requirements().modules) == expected_requirements


@long
def test_libgomp(model):
    req = model.get_requirements()
    assert req.of_type(UnixPackageRequirement) == [
        UnixPackageRequirement(package_name="libgomp1")
    ]
