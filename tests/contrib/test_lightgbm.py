import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from mlem.contrib.lightgbm import (
    LIGHTGBM_DATA,
    LIGHTGBM_LABEL,
    LightGBMDataReader,
    LightGBMDataType,
    LightGBMDataWriter,
    LightGBMModel,
)
from mlem.contrib.numpy import NumpyNdarrayType
from mlem.contrib.pandas import DataFrameType
from mlem.core.artifacts import LOCAL_STORAGE
from mlem.core.data_type import (
    ArrayType,
    DataAnalyzer,
    DataType,
    PrimitiveType,
)
from mlem.core.errors import DeserializationError, SerializationError
from mlem.core.model import Argument, ModelAnalyzer, ModelType
from mlem.core.requirements import UnixPackageRequirement
from tests.conftest import data_write_read_check, long


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
        label=np.array([0, 1]),
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
    assert isinstance(dtype_np.labels, ArrayType)
    assert dtype_np.labels.dtype == PrimitiveType(data=None, ptype="float")
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
    assert isinstance(dtype_df.labels, NumpyNdarrayType)
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
    "lgb_dtype, data_type, label_type",
    [
        ("dtype_np", NumpyNdarrayType, ArrayType),
        ("dtype_df", DataFrameType, NumpyNdarrayType),
    ],
)
def test_lightgbm_source(lgb_dtype, data_type, label_type, request):
    lgb_dtype = request.getfixturevalue(lgb_dtype)
    assert isinstance(lgb_dtype, LightGBMDataType)
    assert isinstance(lgb_dtype.inner, data_type)
    assert isinstance(lgb_dtype.labels, label_type)

    def custom_assert(x, y):
        assert hasattr(x, "data")
        assert hasattr(y, "data")
        assert all(x.data == y.data)
        label_check = x.label == y.label
        if isinstance(label_check, (list, np.ndarray)):
            assert all(label_check)
        else:
            assert label_check

    artifacts = data_write_read_check(
        lgb_dtype,
        writer=LightGBMDataWriter(),
        reader_type=LightGBMDataReader,
        custom_assert=custom_assert,
    )

    if isinstance(lgb_dtype.inner, NumpyNdarrayType):
        assert list(artifacts.keys()) == [
            f"{LIGHTGBM_DATA}/data",
            f"{LIGHTGBM_LABEL}/0/data",
            f"{LIGHTGBM_LABEL}/1/data",
            f"{LIGHTGBM_LABEL}/2/data",
            f"{LIGHTGBM_LABEL}/3/data",
            f"{LIGHTGBM_LABEL}/4/data",
        ]
        assert artifacts[f"{LIGHTGBM_DATA}/data"].uri.endswith(
            f"data/{LIGHTGBM_DATA}"
        )
        assert artifacts[f"{LIGHTGBM_LABEL}/0/data"].uri.endswith(
            f"data/{LIGHTGBM_LABEL}/0"
        )
        assert artifacts[f"{LIGHTGBM_LABEL}/1/data"].uri.endswith(
            f"data/{LIGHTGBM_LABEL}/1"
        )
        assert artifacts[f"{LIGHTGBM_LABEL}/2/data"].uri.endswith(
            f"data/{LIGHTGBM_LABEL}/2"
        )
        assert artifacts[f"{LIGHTGBM_LABEL}/3/data"].uri.endswith(
            f"data/{LIGHTGBM_LABEL}/3"
        )
        assert artifacts[f"{LIGHTGBM_LABEL}/4/data"].uri.endswith(
            f"data/{LIGHTGBM_LABEL}/4"
        )
    else:
        assert list(artifacts.keys()) == [
            f"{LIGHTGBM_DATA}/data",
            f"{LIGHTGBM_LABEL}/data",
        ]
        assert artifacts[f"{LIGHTGBM_DATA}/data"].uri.endswith(
            f"data/{LIGHTGBM_DATA}"
        )
        assert artifacts[f"{LIGHTGBM_LABEL}/data"].uri.endswith(
            f"data/{LIGHTGBM_LABEL}"
        )


def test_serialize__np(dtype_np, np_payload):
    ds = lgb.Dataset(np_payload, label=np_payload.reshape((-1,)).tolist())
    payload = dtype_np.serialize(ds)
    assert payload[LIGHTGBM_DATA] == np_payload.tolist()
    assert payload[LIGHTGBM_LABEL] == np_payload.reshape((-1,)).tolist()

    with pytest.raises(SerializationError):
        dtype_np.serialize({"abc": 123})  # wrong type


def test_deserialize__np(dtype_np, np_payload):
    ds = dtype_np.deserialize(
        {
            LIGHTGBM_DATA: np_payload,
            LIGHTGBM_LABEL: np_payload.reshape((-1,)).tolist(),
        }
    )
    assert isinstance(ds, lgb.Dataset)
    assert np.all(ds.data == np_payload)
    assert np.all(ds.label == np_payload.reshape((-1,)).tolist())

    with pytest.raises(DeserializationError):
        dtype_np.deserialize({LIGHTGBM_DATA: [[1], ["abc"]]})  # illegal matrix


def test_serialize__df(df_payload):
    ds = lgb.Dataset(df_payload, label=None, free_raw_data=False)
    payload = DataType.create(obj=ds)
    assert payload.serialize(ds)["values"] == df_payload.to_dict("records")
    assert LIGHTGBM_LABEL not in payload

    def custom_assert(x, y):
        assert hasattr(x, "data")
        assert hasattr(y, "data")
        assert all(x.data == y.data)
        assert x.label == y.label

    artifacts = data_write_read_check(
        payload,
        writer=LightGBMDataWriter(),
        reader_type=LightGBMDataReader,
        custom_assert=custom_assert,
    )

    assert len(artifacts.keys()) == 1
    assert list(artifacts.keys()) == ["data"]
    assert artifacts["data"].uri.endswith("/data")


def test_deserialize__df(dtype_df, df_payload):
    ds = dtype_df.deserialize(
        {
            LIGHTGBM_DATA: {"values": df_payload},
            LIGHTGBM_LABEL: np.array([0, 1]).tolist(),
        }
    )
    assert isinstance(ds, lgb.Dataset)
    assert ds.data.equals(df_payload)


def test_hook(model, booster, data_np):
    assert isinstance(model, LightGBMModel)
    assert model.model == booster
    assert "predict" in model.methods
    data_type = DataAnalyzer.analyze(data_np)

    signature = model.methods["predict"]
    assert signature.name == "predict"
    assert signature.args[0] == Argument(name="data", type_=data_type)
    returns = NumpyNdarrayType(shape=(None,), dtype="float64")
    assert signature.returns == returns


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
    expected_requirements = {"lightgbm", "numpy"}
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
