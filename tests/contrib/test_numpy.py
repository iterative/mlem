from json import loads

import numpy as np
import pytest
from pydantic import parse_obj_as

from mlem.contrib.numpy import (
    NumpyNdarrayType,
    NumpyNumberType,
    np_type_from_string,
    python_type_from_np_string_repr,
    python_type_from_np_type,
)
from mlem.core.dataset_type import Dataset, DatasetAnalyzer, DatasetType
from mlem.core.errors import DeserializationError, SerializationError
from mlem.utils.module import get_object_requirements
from tests.conftest import dataset_write_read_check


def test_ndarray_source():
    data = np.array([1, 2, 3])
    dataset = Dataset.create(data)
    dataset_write_read_check(dataset, custom_eq=np.array_equal)


@pytest.fixture
def nat():
    return DatasetAnalyzer.analyze(np.array([[1, 2], [3, 4]]))


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
    ndt = DatasetAnalyzer.analyze(value)
    assert isinstance(ndt, NumpyNumberType)
    assert ndt.dtype == "float32"
    assert ndt.get_requirements().modules == ["numpy"]
    payload = {"dtype": "float32", "type": "number"}
    ndt2 = parse_obj_as(DatasetType, payload)
    assert ndt == ndt2
    assert ndt.get_model().__name__ == ndt2.get_model().__name__
    assert ndt.get_model().schema() == {
        "title": "NumpyNumber",
        "type": "number",
    }


def test_ndarray(nat):
    assert isinstance(nat, NumpyNdarrayType)
    assert nat.shape == (None, 2)
    assert python_type_from_np_string_repr(nat.dtype) == int
    assert nat.get_requirements().modules == ["numpy"]
    payload = nat.json()
    nat2 = parse_obj_as(DatasetType, loads(payload))
    assert nat == nat2
    assert nat.get_model().__name__ == nat2.get_model().__name__
    assert nat.get_model().schema() == {
        "title": "NumpyNdarray",
        "type": "array",
        "items": {
            "type": "array",
            "items": {"type": "integer"},
            "minItems": 2,
            "maxItems": 2,
        },
    }


@pytest.mark.parametrize(
    "obj",
    [
        {},  # wrong type
        np.array([[1, 2], [3, 4]], dtype=np.float32),  # wrong data type
        np.array([1, 2]),  # wrong shape
    ],
)
def test_ndarray_serialize_failure(nat, obj):
    with pytest.raises(SerializationError):
        nat.serialize(obj)


@pytest.mark.parametrize(
    "obj",
    [{}, [[1, 2], [3]], [[1, 2, 3]]],  # wrong type  # illegal array  # shape
)
def test_ndarray_deserialize_failure(nat, obj):
    with pytest.raises(DeserializationError):
        nat.deserialize(obj)


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
