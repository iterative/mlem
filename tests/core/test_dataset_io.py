import os

import numpy as np
import pandas as pd
import pytest

from mlem.contrib.numpy import DATA_FILE, NumpyArrayReader, NumpyArrayWriter
from mlem.contrib.pandas import PANDAS_FORMATS, PandasReader, PandasWriter
from mlem.core.artifacts import FSSpecStorage
from mlem.core.dataset_type import Dataset


def test_numpy_read_write():
    data = np.array([1, 2, 3])
    dataset = Dataset.create(data)

    writer = NumpyArrayWriter()
    storage = FSSpecStorage(uri="memory://")
    reader, artifacts = writer.write(dataset, storage, "/")

    assert isinstance(reader, NumpyArrayReader)
    assert len(artifacts) == 1
    assert storage.get_fs().exists(os.path.join("/", DATA_FILE))

    dataset2 = reader.read(artifacts)
    assert isinstance(dataset2, Dataset)
    assert dataset2.dataset_type == dataset.dataset_type
    assert isinstance(dataset2.data, np.ndarray)
    assert np.array_equal(dataset2.data, data)


@pytest.mark.parametrize("format", list(PANDAS_FORMATS.keys()))
def test_pandas_read_write(format):
    data = pd.DataFrame([{"a": 1, "b": 2}])
    dataset = Dataset.create(data)
    storage = FSSpecStorage(uri="memory://")

    writer = PandasWriter(format=format)
    reader, artifacts = writer.write(dataset, storage, "/")

    assert isinstance(reader, PandasReader)
    filename = writer.fmt.file_name
    assert len(artifacts) == 1
    assert storage.get_fs().exists(os.path.join("/", filename))

    dataset2 = reader.read(artifacts)
    assert isinstance(dataset2, Dataset)
    assert dataset2.dataset_type == dataset.dataset_type
    assert isinstance(dataset2.data, pd.DataFrame)

    assert dataset2.data.equals(data)
