import numpy as np
import pytest
from scipy.sparse import csr_matrix

from mlem.contrib.scipy import ScipySparseMatrix
from mlem.core.data_type import DataAnalyzer
from tests.conftest import data_write_read_check


@pytest.fixture
def test_data():
    row = np.array([0, 0, 1, 2, 2, 2])
    col = np.array([0, 2, 2, 0, 1, 2])
    data = np.array([1, 2, 3, 4, 5, 6])
    return csr_matrix((data, (row, col)), shape=(3, 3), dtype="float32")


def test_sparce_matrix(test_data):
    assert ScipySparseMatrix.is_object_valid(test_data)
    sdt = DataAnalyzer.analyze(test_data)
    assert sdt.dict() == {"dtype": "float32", "type": "csr_matrix"}
    assert isinstance(sdt, ScipySparseMatrix)
    assert sdt.dtype == "float32"
    assert sdt.get_requirements().modules == ["scipy"]


def test_write_read(test_data):
    sdt = DataAnalyzer.analyze(test_data)
    sdt = sdt.bind(test_data)
    data_write_read_check(
        sdt, custom_eq=lambda x, y: np.array_equal(x.todense(), y.todense())
    )
