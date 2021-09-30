import numpy as np
import pandas as pd
import pytest
from pydantic import parse_obj_as

from mlem.contrib.numpy import NumpyNdarrayType, NumpyNumberType
from mlem.contrib.pandas import DataFrameType
from mlem.core.dataset_type import DatasetAnalyzer, DatasetType, PrimitiveType


class NotPrimitive:
    pass


def test_primitives_not_ok():
    assert not PrimitiveType.is_object_valid(NotPrimitive())


@pytest.mark.parametrize("ptype", PrimitiveType.PRIMITIVES)
def test_primitives(ptype):
    value = ptype()
    assert PrimitiveType.is_object_valid(value)
    dt = DatasetAnalyzer.analyze(value)
    assert isinstance(dt, PrimitiveType)
    assert dt.ptype == ptype.__name__
    payload = {"ptype": ptype.__name__, "type": "primitive"}
    assert dt.dict() == payload
    dt2 = parse_obj_as(DatasetType, payload)
    assert dt2 == dt


def test_numpy_number():
    value = np.int8(0)
    assert NumpyNumberType.is_object_valid(value)
    dt = DatasetAnalyzer.analyze(value)
    assert isinstance(dt, NumpyNumberType)
    assert dt.dtype == "int8"
    payload = {"dtype": "int8", "type": "number"}
    assert dt.dict() == payload
    dt2 = parse_obj_as(DatasetType, payload)
    assert dt2 == dt


def test_numpy_ndarray():
    value = np.zeros((1, 1), "int8")
    assert NumpyNdarrayType.is_object_valid(value)
    dt = DatasetAnalyzer.analyze(value)
    assert isinstance(dt, NumpyNdarrayType)
    assert dt.shape == (None, 1)
    assert dt.dtype == "int8"
    payload = {"type": "ndarray", "shape": (None, 1), "dtype": "int8"}
    assert dt.dict() == payload
    dt2 = parse_obj_as(DatasetType, payload)
    assert dt2 == dt


# def test_pandas_series():
#     value = pd.Series([1, 2])
#     assert SeriesType.is_object_valid(value)
#     dt = DatasetAnalyzer.analyze(value)
#     assert isinstance(dt, SeriesType)
#     assert dt.columns == ['a']
#     assert dt.dtypes == ['int64']
#     assert dt.index_cols == []


def test_pandas_dataframe():
    value = pd.DataFrame([{"a": 1}])
    assert DataFrameType.is_object_valid(value)
    dt = DatasetAnalyzer.analyze(value)
    assert isinstance(dt, DataFrameType)
    assert dt.columns == ["a"]
    assert dt.dtypes == ["int64"]
    assert dt.index_cols == []
    payload = {
        "type": "dataframe",
        "columns": ["a"],
        "dtypes": ["int64"],
        "index_cols": [],
    }
    assert dt.dict() == payload
    dt2 = parse_obj_as(DatasetType, payload)
    assert dt2 == dt
