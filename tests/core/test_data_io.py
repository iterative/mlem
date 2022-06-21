import numpy as np
import pandas as pd
import pytest

from mlem.contrib.numpy import NumpyArrayReader, NumpyArrayWriter
from mlem.contrib.pandas import PANDAS_FORMATS, PandasReader, PandasWriter
from mlem.core.artifacts import FSSpecStorage
from mlem.core.data_type import DataType


def test_numpy_read_write():
    data = np.array([1, 2, 3])
    data_type = DataType.create(data)

    writer = NumpyArrayWriter()
    storage = FSSpecStorage(uri="memory://")
    reader, artifacts = writer.write(data_type, storage, "/data")

    assert isinstance(reader, NumpyArrayReader)
    assert len(artifacts) == 1
    assert storage.get_fs().exists("/data")

    data_type2 = reader.read(artifacts)
    assert isinstance(data_type2, DataType)
    assert data_type2 == data_type
    assert isinstance(data_type2.data, np.ndarray)
    assert np.array_equal(data_type2.data, data)


@pytest.mark.parametrize("format", list(PANDAS_FORMATS.keys()))
def test_pandas_read_write(format):
    data = pd.DataFrame([{"a": 1, "b": 2}])
    data_type = DataType.create(data)
    storage = FSSpecStorage(uri="memory://")

    writer = PandasWriter(format=format)
    reader, artifacts = writer.write(data_type, storage, "/data")

    assert isinstance(reader, PandasReader)
    assert len(artifacts) == 1
    assert storage.get_fs().exists("/data")

    data_type2 = reader.read(artifacts)
    assert isinstance(data_type2, DataType)
    assert data_type2 == data_type
    assert isinstance(data_type2.data, pd.DataFrame)

    assert data_type2.data.equals(data)
