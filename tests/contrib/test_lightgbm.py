import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from mlem.contrib.lightgbm import (
    LightGBMDataReader,
    LightGBMDataType,
    LightGBMDataWriter,
    LightGBMModel,
)
from mlem.contrib.numpy import NumpyNdarrayType
from mlem.contrib.pandas import DataFrameType
from mlem.core.artifacts import LOCAL_STORAGE
from mlem.core.data_type import DataAnalyzer, DataType
from mlem.core.errors import DeserializationError, SerializationError
from mlem.core.model import ModelAnalyzer, ModelType
from mlem.core.requirements import UnixPackageRequirement
from tests.conftest import (
    check_model_type_common_interface,
    data_write_read_check,
    long,
)


@pytest.fixture
def np_payload():
    return np.linspace(0, 2, 5).reshape((-1, 1))


@pytest.fixture
def data_np(np_payload):
    return lgb.Dataset(
        np_payload,
        label=np_payload.reshape((-1,)).tolist(),
        free_raw_data=False,
    )


@pytest.fixture
def df_payload():
    return pd.DataFrame([{"a": i} for i in range(2)])


@pytest.fixture
def data_df(df_payload):
    return lgb.Dataset(
        df_payload,
        label=np.array([0, 1]).tolist(),
        free_raw_data=False,
    )


@pytest.fixture
def booster(data_np):
    return lgb.train({}, data_np, 1)


@pytest.fixture
def model(booster, data_np) -> ModelType:
    return ModelAnalyzer.analyze(booster, sample_data=data_np)


@pytest.fixture
def dtype_np(data_np):
    return DataType.create(obj=data_np)


@pytest.fixture
def dtype_df(data_df):
    return DataType.create(obj=data_df)


def test_hook_np(dtype_np: DataType):
    assert set(dtype_np.get_requirements().modules) == {"lightgbm", "numpy"}
    assert isinstance(dtype_np, LightGBMDataType)
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


def test_hook_df(dtype_df: DataType):
    assert set(dtype_df.get_requirements().modules) == {"lightgbm", "pandas"}
    assert isinstance(dtype_df, LightGBMDataType)
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


@pytest.mark.parametrize(
    "lgb_dtype, data_type",
    [("dtype_np", NumpyNdarrayType), ("dtype_df", DataFrameType)],
)
def test_lightgbm_source(lgb_dtype, data_type, request):
    lgb_dtype = request.getfixturevalue(lgb_dtype)
    assert isinstance(lgb_dtype, LightGBMDataType)
    assert isinstance(lgb_dtype.inner, data_type)

    def custom_assert(x, y):
        assert hasattr(x, "data")
        assert hasattr(y, "data")
        assert all(x.data == y.data)
        assert all(x.label == y.label)

    data_write_read_check(
        lgb_dtype,
        writer=LightGBMDataWriter(),
        reader_type=LightGBMDataReader,
        custom_assert=custom_assert,
    )


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


def test_hook(model, booster, data_np):
    assert isinstance(model, LightGBMModel)
    assert model.model == booster
    assert "lightgbm_predict" in model.methods
    data_type = DataAnalyzer.analyze(data_np)

    check_model_type_common_interface(
        model, data_type, NumpyNdarrayType(shape=(None,), dtype="float64")
    )


def test_model__predict(model, data_np):
    predict = model.call_method("predict", data_np)
    assert isinstance(predict, np.ndarray)
    assert len(predict) == data_np.num_data()


def test_model__predict_not_dataset(model):
    data = [[1]]
    predict = model.call_method("predict", data)
    assert isinstance(predict, np.ndarray)
    assert len(predict) == len(data)


@long
def test_model__dump_load(tmpdir, model, data_np, local_fs):
    # pandas is not required, but if it is installed, it is imported by lightgbm
    expected_requirements = {"lightgbm", "numpy", "scipy", "pandas"}
    assert set(model.get_requirements().modules) == expected_requirements

    artifacts = model.dump(LOCAL_STORAGE, tmpdir)

    model.unbind()
    with pytest.raises(ValueError):
        model.call_method("predict", data_np)

    model.load(artifacts)
    test_model__predict(model, data_np)

    assert set(model.get_requirements().modules) == expected_requirements


@long
def test_libgomp(model):
    req = model.get_requirements()
    assert req.of_type(UnixPackageRequirement) == [
        UnixPackageRequirement(package_name="libgomp1")
    ]
