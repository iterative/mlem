import io
import json
from datetime import datetime, timezone
from typing import Callable, List, Union

import pandas as pd
import pytest

from mlem.contrib.pandas import (
    PANDAS_FORMATS,
    DataFrameType,
    PandasReader,
    PandasWriter,
    pd_type_from_string,
    python_type_from_pd_string_repr,
    python_type_from_pd_type,
    string_repr_from_pd_type,
)
from mlem.core.dataset_type import Dataset, DatasetAnalyzer, DatasetType
from mlem.core.errors import DeserializationError, SerializationError
from mlem.core.meta_io import deserialize, serialize
from tests.conftest import dataset_write_read_check

PD_DATA_FRAME = pd.DataFrame(
    [
        {
            "int": 1,
            "str": "a",
            "float": 0.1,
            "dt": datetime.now(),
            "bool": True,
            "dt_tz": datetime.now(timezone.utc),
        },
        {
            "int": 2,
            "str": "b",
            "float": 0.2,
            "dt": datetime.now(),
            "bool": False,
            "dt_tz": datetime.now(timezone.utc),
        },
    ]
)

PD_DATA_FRAME_INDEX = PD_DATA_FRAME.set_index("int")
PD_DATA_FRAME_MULTIINDEX = PD_DATA_FRAME.set_index(["int", "str"])


def df_to_str(df: pd.DataFrame):
    buf = io.StringIO()
    df.to_csv(buf)
    return buf.getvalue()


def pandas_assert(actual: pd.DataFrame, expected: pd.DataFrame):
    assert list(expected.columns) == list(actual.columns), "different columns"
    assert expected.shape == actual.shape, "different shapes"
    assert list(expected.dtypes) == list(actual.dtypes), "different dtypes"
    assert list(expected.index) == list(actual.index), "different indexes"
    assert df_to_str(expected) == df_to_str(
        actual
    ), "different str representation"
    assert expected.equals(actual), "contents are not equal"


@pytest.fixture
def data():
    return pd.DataFrame([{"a": 1, "b": 3, "c": 5}, {"a": 2, "b": 4, "c": 6}])


@pytest.fixture
def data2():
    return PD_DATA_FRAME


@pytest.fixture
def df_type(data):
    return DatasetAnalyzer.analyze(data)


@pytest.fixture
def df_type2(data2):
    return DatasetAnalyzer.analyze(data2)


@pytest.fixture
def series_data(data2):
    return data2.iloc[0]


@pytest.fixture
def series_df_type(series_data):
    return DatasetAnalyzer.analyze(series_data)


def for_all_formats(exclude: Union[List[str], Callable] = None):
    ex = exclude if isinstance(exclude, list) else []
    formats = [name for name in PANDAS_FORMATS if name not in ex]
    mark = pytest.mark.parametrize("format", formats)
    if isinstance(exclude, list):
        return mark
    return mark(exclude)


@for_all_formats
def test_simple_df(data, format):
    writer = PandasWriter(format=format)
    dataset_write_read_check(
        Dataset.create(data), writer, PandasReader, pd.DataFrame.equals
    )


@for_all_formats
def test_with_index(data, format):
    writer = PandasWriter(format=format)
    dataset_write_read_check(
        Dataset.create(data.set_index("a")),
        writer,
        PandasReader,
        custom_assert=pandas_assert,
    )


@for_all_formats
def test_with_multiindex(data, format):
    writer = PandasWriter(format=format)
    dataset_write_read_check(
        Dataset.create(data.set_index(["a", "b"])),
        writer,
        PandasReader,
        custom_assert=pandas_assert,
    )


@for_all_formats(
    exclude=[
        "excel",  # Excel does not support datetimes with timezones
        "parquet",  # Casting from timestamp[ns] to timestamp[ms] would lose data
    ]
)
@pytest.mark.parametrize(
    "data", [PD_DATA_FRAME, PD_DATA_FRAME_INDEX, PD_DATA_FRAME_MULTIINDEX]
)
def test_with_index_complex(data, format):
    writer = PandasWriter(format=format)
    dataset_write_read_check(
        Dataset.create(data), writer, PandasReader, custom_assert=pandas_assert
    )


for_all_dtypes = pytest.mark.parametrize(
    "dtype", PD_DATA_FRAME.dtypes, ids=[str(d) for d in PD_DATA_FRAME.dtypes]
)


@for_all_dtypes
def test_string_repr_from_pd_type(dtype):
    assert isinstance(string_repr_from_pd_type(dtype), str)


@for_all_dtypes
def test_python_type_from_pd_type(dtype):
    python_type_from_pd_type(dtype)


@for_all_dtypes
def test_pd_type_from_string(dtype):
    pd_type_from_string(string_repr_from_pd_type(dtype))


@for_all_dtypes
def test_python_type_from_pd_string_repr(dtype):
    python_type_from_pd_string_repr(string_repr_from_pd_type(dtype))


@pytest.mark.parametrize("df_type_fx", ["df_type2", "df_type"])
def test_df_type(df_type_fx, request):
    df_type = request.getfixturevalue(df_type_fx)
    assert isinstance(df_type, DataFrameType)

    obj = serialize(df_type)
    new_df_type = deserialize(obj, DatasetType)

    assert df_type == new_df_type


def test_dataframe_type(df_type: DataFrameType, data):
    assert df_type.get_requirements().modules == ["pandas"]

    obj = df_type.serialize(data)
    payload = json.dumps(obj)
    loaded = json.loads(payload)
    data2 = df_type.deserialize(loaded)

    assert data.equals(data2)


@pytest.mark.parametrize(
    "obj",
    [
        {"a": [1, 2], "b": [1, 2]},  # not a dataframe
        pd.DataFrame([{"a": 1}, {"a": 2}]),  # wrong columns
    ],
)
def test_dataframe_serialize_failure(df_type, obj):
    with pytest.raises(SerializationError):
        df_type.serialize(obj)


@pytest.mark.parametrize(
    "obj",
    [
        1,  # not a dict
        {},  # no `values` key
        {"values": [{"a": 1}, {"a": 2}]},  # wrong columns
    ],
)
def test_dataframe_deserialize_failure(df_type: DataFrameType, obj):
    with pytest.raises(DeserializationError):
        df_type.deserialize(obj)


def test_unordered_columns(df_type: DataFrameType, data):
    data_rev = data[list(reversed(data.columns))]
    obj = df_type.serialize(data_rev)
    data2 = df_type.deserialize(obj)

    assert data.equals(data2), f"{data} \n!=\n{data2}"
    assert data2 is not data


def test_datetime():
    data = pd.DataFrame(
        [{"a": 1, "b": datetime.now()}, {"a": 2, "b": datetime.now()}]
    )
    df_type = DatasetAnalyzer.analyze(data)
    assert isinstance(df_type, DataFrameType)

    obj = df_type.serialize(data)
    payload = json.dumps(obj)
    loaded = json.loads(payload)
    data2 = df_type.deserialize(loaded)

    assert data.equals(data2)
    assert data2 is not data


@pytest.mark.parametrize(
    "df", [PD_DATA_FRAME, PD_DATA_FRAME_INDEX, PD_DATA_FRAME_MULTIINDEX]
)
def test_all(df):
    df_type: DataFrameType = DatasetAnalyzer.analyze(df)

    obj = df_type.serialize(df)
    payload = json.dumps(obj)
    loaded = json.loads(payload)
    data = df_type.deserialize(loaded)

    assert df is not data
    pandas_assert(data, df)


@pytest.mark.parametrize(
    "df", [PD_DATA_FRAME, PD_DATA_FRAME_INDEX, PD_DATA_FRAME_MULTIINDEX]
)
def test_all_filtered(df):
    df = df[~df["bool"]]
    df_type: DataFrameType = DatasetAnalyzer.analyze(df)

    obj = df_type.serialize(df)
    payload = json.dumps(obj)
    loaded = json.loads(payload)
    data = df_type.deserialize(loaded)

    assert df is not data
    pandas_assert(data, df)


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
