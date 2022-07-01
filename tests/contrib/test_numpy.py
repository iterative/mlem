import re

import numpy as np
import pytest
from pydantic import parse_obj_as
from pytest_lazyfixture import lazy_fixture

from mlem.contrib.numpy import (
    NumpyNdarrayType,
    NumpyNumberReader,
    NumpyNumberType,
    np_type_from_string,
    python_type_from_np_string_repr,
    python_type_from_np_type,
)
from mlem.core.data_type import DataAnalyzer, DataType
from mlem.core.errors import DeserializationError, SerializationError
from mlem.utils.module import get_object_requirements
from tests.conftest import data_write_read_check


def test_npnumber_source():
    data = np.float32(1.5)
    data_type = DataType.create(data)

    def custom_assert(x, y):
        assert x.dtype == y.dtype
        assert isinstance(x, np.number)
        assert isinstance(y, np.number)
        assert x.dtype.name == data.dtype.name == y.dtype.name

    data_write_read_check(
        data_type,
        custom_eq=np.equal,
        reader_type=NumpyNumberReader,
        custom_assert=custom_assert,
    )


@pytest.fixture
def nat(numpy_default_int_dtype):
    data = np.array([[1, 2], [3, 4]])
    dtype = DataType.create(data)
    payload = {
        "shape": (None, 2),
        "dtype": numpy_default_int_dtype,
        "type": "ndarray",
    }
    schema = {
        "title": "NumpyNdarray",
        "type": "array",
        "items": {
            "type": "array",
            "items": {"type": "integer"},
            "minItems": 2,
            "maxItems": 2,
        },
    }
    test_data1 = data
    test_data2 = np.array([[10, 20], [30, 40]])
    test_data3 = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    return False, dtype, payload, schema, test_data1, test_data2, test_data3


@pytest.fixture
def nat_dynamic():
    dtype = NumpyNdarrayType(shape=[2, None, None], dtype="int")
    payload = {"dtype": "int", "shape": (2, None, None), "type": "ndarray"}
    schema = {
        "items": {
            "items": {"items": {"type": "integer"}, "type": "array"},
            "type": "array",
        },
        "maxItems": 2,
        "minItems": 2,
        "title": "NumpyNdarray",
        "type": "array",
    }
    test_data1 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    test_data2 = np.array(
        [[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]]
    )
    test_data3 = np.array([[[1, 2, 2], [3, 4, 4]], [[5, 6, 6], [7, 8, 8]]])
    return True, dtype, payload, schema, test_data1, test_data2, test_data3


@pytest.fixture
def nat_dynamic_float():
    dtype = NumpyNdarrayType(shape=[2, None, None, 1], dtype="float")
    payload = {
        "dtype": "float",
        "shape": (2, None, None, 1),
        "type": "ndarray",
    }
    schema = {
        "items": {
            "items": {
                "items": {
                    "items": {"type": "number"},
                    "maxItems": 1,
                    "minItems": 1,
                    "type": "array",
                },
                "type": "array",
            },
            "type": "array",
        },
        "maxItems": 2,
        "minItems": 2,
        "title": "NumpyNdarray",
        "type": "array",
    }
    test_data1 = np.array([[[[1.0]], [[3.0]]], [[[5.1]], [[7.1]]]])
    test_data2 = np.array([[[[1.1], [3.0], [5.0]]], [[[7.1], [9.99], [11.2]]]])
    test_data3 = np.array(
        [[[[1.1], [3.2]], [[5.33], [7.1]]], [[[1.11], [3.4]], [[5.3], [7.2]]]]
    )
    return True, dtype, payload, schema, test_data1, test_data2, test_data3


@pytest.fixture
def nat_dynamic_all_none_dims():
    dtype = NumpyNdarrayType(shape=[None, None, None], dtype="int")
    payload = {"dtype": "int", "shape": (None, None, None), "type": "ndarray"}
    schema = {
        "items": {
            "items": {"items": {"type": "integer"}, "type": "array"},
            "type": "array",
        },
        "title": "NumpyNdarray",
        "type": "array",
    }
    test_data1 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    test_data2 = np.array(
        [[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]]
    )
    test_data3 = np.array([[[1, 2, 2], [3, 4, 4]], [[5, 6, 6], [7, 8, 8]]])
    return True, dtype, payload, schema, test_data1, test_data2, test_data3


@pytest.fixture
def nat_dynamic_shape_none():
    dtype = NumpyNdarrayType(shape=None, dtype="int")
    payload = {"dtype": "int", "type": "ndarray"}
    schema = {
        "items": {
            "anyOf": [{"type": "integer"}, {"items": {}, "type": "array"}]
        },
        "title": "NumpyNdarray",
        "type": "array",
    }
    test_data1 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    test_data2 = np.array(
        [[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]]
    )
    test_data3 = np.array([[[1, 2, 2], [3, 4, 4]], [[5, 6, 6], [7, 8, 8]]])
    return True, dtype, payload, schema, test_data1, test_data2, test_data3


@pytest.fixture
def nat_shape_empty():
    dtype = NumpyNdarrayType(shape=(), dtype="int")
    payload = {"dtype": "int", "shape": (), "type": "ndarray"}
    schema = {
        "title": "NumpyNdarray",
        "type": "integer",
    }
    test_data1 = np.array(1)
    test_data2 = np.array(3)
    test_data3 = np.array(4)
    return True, dtype, payload, schema, test_data1, test_data2, test_data3


def test_python_type_from_np_string_repr():
    assert python_type_from_np_string_repr("int64") == int

    with pytest.raises(ValueError):
        python_type_from_np_string_repr("int65")


def test_python_type_from_np_type():
    assert python_type_from_np_type(np.dtype(np.int64)) == int


def test_np_type_from_string():
    assert isinstance(np_type_from_string("int64"), np.dtype)

    with pytest.raises(ValueError):
        np_type_from_string("int65")


def test_number():
    value = np.float32(0.5)
    assert NumpyNumberType.is_object_valid(value)
    ndt = DataAnalyzer.analyze(value)
    assert isinstance(ndt, NumpyNumberType)
    assert ndt.dtype == "float32"
    assert ndt.get_requirements().modules == ["numpy"]
    payload = {"dtype": "float32", "type": "number"}
    ndt2 = parse_obj_as(DataType, payload)
    assert ndt == ndt2
    assert ndt.get_model().__name__ == ndt2.get_model().__name__
    assert ndt.get_model() is float
    n_payload = ndt.get_serializer().serialize(value)
    assert ndt.get_serializer().deserialize(n_payload) == value


@pytest.mark.parametrize("test_data_idx", [4, 5, 6])
@pytest.mark.parametrize(
    "data",
    [
        (lazy_fixture("nat")),
        (lazy_fixture("nat_dynamic")),
        (lazy_fixture("nat_dynamic_all_none_dims")),
        (lazy_fixture("nat_dynamic_shape_none")),
        (lazy_fixture("nat_dynamic_float")),
        (lazy_fixture("nat_shape_empty")),
    ],
)
def test_ndarray(data, test_data_idx):
    nat, payload, schema, value = (
        data[1],
        data[2],
        data[3],
        data[test_data_idx],
    )
    assert isinstance(nat, NumpyNdarrayType)
    assert nat.get_requirements().modules == ["numpy"]
    assert nat.dict() == payload
    nat2 = parse_obj_as(DataType, payload)
    assert nat == nat2
    assert nat.get_model().__name__ == nat2.get_model().__name__
    assert nat.get_model().schema() == schema
    n_payload = nat.get_serializer().serialize(value)
    assert (nat.get_serializer().deserialize(n_payload) == value).all()
    model = parse_obj_as(nat.get_model(), n_payload)
    assert model.__root__ == n_payload

    nat = nat.bind(value)
    data_write_read_check(nat, custom_eq=np.array_equal)


@pytest.mark.parametrize(
    "nddtype,obj,err_msg",
    [
        [
            lazy_fixture("nat"),
            {},
            "given data is of type: <class 'dict'>, expected: <class 'numpy.ndarray'>",
        ],
        [
            lazy_fixture("nat"),
            np.array([[1, 2], [3, 4]], dtype=np.float32),
            f"given array is of type: float32, expected: {np.array([[1, 2], [3, 4]]).dtype}",
        ],
        [
            lazy_fixture("nat"),
            np.array([1, 2]),
            "given array is of rank: 1, expected: 2",
        ],
        [
            lazy_fixture("nat_dynamic"),
            np.array([1, 2]),
            "given array is of rank: 1, expected: 3",
        ],
        [
            lazy_fixture("nat_shape_empty"),
            np.array([1, 2]),
            "given array is of rank: 1, expected: 0",
        ],
        [
            lazy_fixture("nat_dynamic_float"),
            np.array([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]),
            "given array is of shape: (1, None, None, 2), expected: (2, None, None, 1)",
        ],
    ],
)
def test_ndarray_serialize_failure(nddtype, obj, err_msg):
    with pytest.raises(SerializationError, match=re.escape(err_msg)):
        nddtype[1].serialize(obj)


@pytest.mark.parametrize(
    "obj",
    [{}, [[1, 2], [3]], [[1, 2, 3]]],  # wrong type  # illegal array  # shape
)
def test_ndarray_deserialize_failure(nat, obj):
    with pytest.raises(DeserializationError):
        nat[1].deserialize(obj)


def test_requirements():
    assert get_object_requirements(
        NumpyNdarrayType(shape=(0,), dtype="int")
    ).modules == ["numpy"]


# Copyright 2019 Zyfra
# Copyright 2021 Iterative
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
