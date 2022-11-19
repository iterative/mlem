import numpy as np
import pandas as pd
import pytest
import xgboost

from mlem.contrib.numpy import NumpyNdarrayType
from mlem.contrib.xgboost import DMatrixDataType, XGBoostModel
from mlem.core.artifacts import LOCAL_STORAGE
from mlem.core.data_type import DataAnalyzer
from mlem.core.errors import DeserializationError, SerializationError
from mlem.core.model import Argument, ModelAnalyzer, ModelType
from mlem.core.requirements import UnixPackageRequirement
from tests.conftest import long


@pytest.fixture
def np_payload():
    return np.linspace(0, 2).reshape((-1, 1))


@pytest.fixture
def dmatrix_np(np_payload):
    return xgboost.DMatrix(np_payload, label=np_payload)


@pytest.fixture
def df_payload():
    return pd.DataFrame([{"a": i} for i in range(2)])


@pytest.fixture
def dmatrix_df(df_payload):
    return xgboost.DMatrix(
        df_payload, label=np.linspace(0, 2).reshape((-1, 1))
    )


@pytest.fixture
def booster(dmatrix_np):
    return xgboost.train({}, dmatrix_np, 1)


@pytest.fixture
def model(booster, np_payload) -> ModelType:
    return ModelAnalyzer.analyze(booster, sample_data=np_payload)


@pytest.fixture
def dtype_np(dmatrix_np):
    return DataAnalyzer.analyze(dmatrix_np)


@pytest.fixture
def dtype_df(dmatrix_df):
    return DataAnalyzer.analyze(dmatrix_df)


def test_hook_np(dtype_np):
    assert isinstance(dtype_np, DMatrixDataType)
    assert dtype_np.get_requirements().modules == ["xgboost"]
    assert dtype_np.is_from_list


def test_hook_df(dtype_df):
    assert isinstance(dtype_df, DMatrixDataType)
    assert dtype_df.get_requirements().modules == ["xgboost"]
    assert not dtype_df.is_from_list
    assert dtype_df.feature_names == ["a"]


def test_serialize__np(dtype_np, np_payload):
    dmatrix = xgboost.DMatrix(np_payload)
    with pytest.raises(SerializationError):
        dtype_np.serialize(dmatrix)


def test_deserialize__np(dtype_np, np_payload):
    dmatrix = dtype_np.deserialize(np_payload)
    assert isinstance(dmatrix, xgboost.DMatrix)


@pytest.mark.parametrize("obj", [[123, "abc"], {"abc": 123}])
def test_deserialize__np_failure(dtype_np, obj):
    with pytest.raises(DeserializationError):
        dtype_np.deserialize(obj)


def test_deserialize__df(dtype_df, df_payload):
    dmatrix = dtype_df.deserialize(df_payload)
    assert isinstance(dmatrix, xgboost.DMatrix)


# def test_np__schema(dtype_np):
#     schema = spec.type_to_schema(dtype_np)
#
#     assert schema == {
#         'items': {'type': 'number'},
#         'maxItems': 1,
#         'minItems': 1,
#         'type': 'array'
#     }
#
#
# def test_df__schema(dtype_df):
#     schema = spec.type_to_schema(dtype_df)
#     assert schema == {'properties': {'a': {'type': 'integer'}}, 'required': ['a'], 'type': 'object'}


def test_hook(model, booster, np_payload):
    assert isinstance(model, XGBoostModel)
    assert model.model == booster

    data_type = DataAnalyzer.analyze(np_payload)
    assert "predict" in model.methods
    signature = model.methods["predict"]
    assert signature.name == "predict"
    assert signature.args[0] == Argument(name="data", type_=data_type)
    returns = NumpyNdarrayType(shape=(None,), dtype="float32")
    assert signature.returns == returns


def test_model__predict(model, dmatrix_np):
    predict = model.call_method("predict", dmatrix_np)
    assert isinstance(predict, np.ndarray)
    assert len(predict) == dmatrix_np.num_row()


def test_model__predict_not_dmatrix(model):
    data = np.asarray([[1]])
    predict = model.call_method("predict", data)
    assert isinstance(predict, np.ndarray)
    assert len(predict) == len(data)


@long
def test_model__dump_load(tmpdir, model, dmatrix_np, local_fs):
    expected_requirements = {"xgboost", "numpy"}
    assert set(model.get_requirements().modules) == expected_requirements

    artifacts = model.dump(LOCAL_STORAGE, tmpdir)
    model.unbind()
    with pytest.raises(ValueError):
        model.call_method("predict", dmatrix_np)

    model.load(artifacts)
    test_model__predict(model, dmatrix_np)

    assert set(model.get_requirements().modules) == expected_requirements


@long
def test_libgomp(model):
    req = model.get_requirements()
    assert req.of_type(UnixPackageRequirement) == [
        UnixPackageRequirement(package_name="libgomp1")
    ]
