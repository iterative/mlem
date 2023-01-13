import numpy as np
import pytest
from scipy.sparse import csr_matrix

from mlem.contrib.scipy import ScipySparseMatrix
from mlem.core.data_type import DataAnalyzer
from mlem.core.errors import DeserializationError, SerializationError
from tests.conftest import data_write_read_check


@pytest.fixture
def raw_data():
    row = np.array([0, 0, 1, 2, 2, 2])
    col = np.array([0, 2, 2, 0, 1, 2])
    data = np.array([1, 2, 3, 4, 5, 6])
    return data, (row, col)


@pytest.fixture
def sparse_mat(raw_data):
    return csr_matrix(raw_data, shape=(3, 3), dtype="float32")


@pytest.fixture
def schema():
    return {
        "title": "ScipySparse",
        "type": "array",
        "items": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 3,
            "maxItems": 3,
        },
    }


@pytest.fixture
def sparse_data_type(sparse_mat):
    return DataAnalyzer.analyze(sparse_mat)


def test_sparce_matrix(sparse_mat, schema):
    assert ScipySparseMatrix.is_object_valid(sparse_mat)
    sdt = DataAnalyzer.analyze(sparse_mat)
    assert sdt.dict() == {
        "dtype": "float32",
        "type": "csr_matrix",
        "shape": (3, 3),
    }
    model = sdt.get_model()
    assert model.__name__ == "ScipySparse"
    assert model.schema() == schema
    assert isinstance(sdt, ScipySparseMatrix)
    assert sdt.dtype == "float32"
    assert sdt.get_requirements().modules == ["scipy"]


def test_serialization(raw_data, sparse_mat):
    sdt = DataAnalyzer.analyze(sparse_mat)
    payload = sdt.serialize(sparse_mat)
    deserialized_data = sdt.deserialize(payload)
    assert np.array_equal(sparse_mat.todense(), deserialized_data.todense())


def test_write_read(sparse_mat):
    sdt = DataAnalyzer.analyze(sparse_mat)
    sdt = sdt.bind(sparse_mat)
    data_write_read_check(
        sdt, custom_eq=lambda x, y: np.array_equal(x.todense(), y.todense())
    )


@pytest.mark.parametrize(
    "obj",
    [
        1,  # wrong type
        csr_matrix(
            ([1], ([1], [0])), shape=(3, 3), dtype="float64"
        ),  # wrong dtype
        csr_matrix(
            ([1], ([1], [0])), shape=(2, 2), dtype="float32"
        ),  # wrong shape
    ],
)
def test_serialize_failure(sparse_mat, obj):
    sdt = DataAnalyzer.analyze(sparse_mat)
    with pytest.raises(SerializationError):
        sdt.serialize(obj)


@pytest.mark.parametrize(
    "obj", [1, ([1, 1], ([0, 6], [1, 6]))]  # wrong type  # wrong shape
)
def test_desiarilze_failure(sparse_data_type, obj):
    with pytest.raises(DeserializationError):
        sparse_data_type.deserialize(obj)
