import numpy as np
import pandas as pd
import pytest
import xgboost

from mlem.contrib.xgboost import DMatrixDatasetType, XGBoostModel
from mlem.core.dataset_type import DatasetAnalyzer
from mlem.core.errors import DeserializationError, SerializationError
from mlem.core.model import ModelAnalyzer, ModelType
from mlem.core.requirements import UnixPackageRequirement


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
def model(booster, dmatrix_np) -> ModelType:
    return ModelAnalyzer.analyze(booster, test_data=dmatrix_np)


@pytest.fixture
def dtype_np(dmatrix_np):
    return DatasetAnalyzer.analyze(dmatrix_np)


@pytest.fixture
def dtype_df(dmatrix_df):
    return DatasetAnalyzer.analyze(dmatrix_df)


def test_hook_np(dtype_np):
    assert isinstance(dtype_np, DMatrixDatasetType)
    assert dtype_np.get_requirements().modules == ["xgboost"]
    assert dtype_np.is_from_list


def test_hook_df(dtype_df):
    assert isinstance(dtype_df, DMatrixDatasetType)
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


def test_hook(model, booster):
    assert isinstance(model, XGBoostModel)
    assert model.model == booster


def test_model__predict(model, dmatrix_np):
    predict = model.call_method("predict", dmatrix_np)
    assert isinstance(predict, np.ndarray)
    assert len(predict) == dmatrix_np.num_row()


def test_model__predict_not_dmatrix(model):
    data = np.asarray([[1]])
    predict = model.call_method("predict", data)
    assert isinstance(predict, np.ndarray)
    assert len(predict) == len(data)


def test_model__dump_load(tmpdir, model, dmatrix_np, local_fs):
    expected_requirements = {"xgboost"}  # , 'numpy'}
    # TODO: https://github.com/iterative/mlem/issues/21 methods
    assert set(model.get_requirements().modules) == expected_requirements

    model.dump(local_fs, tmpdir)
    model.unbind()
    with pytest.raises(ValueError):
        model.call_method("predict", dmatrix_np)

    model.load(local_fs, tmpdir)
    test_model__predict(model, dmatrix_np)

    assert set(model.get_requirements().modules) == expected_requirements


def test_libgomp(model):
    req = model.get_requirements()
    assert req.of_type(UnixPackageRequirement) == [
        UnixPackageRequirement(package_name="libgomp1")
    ]
