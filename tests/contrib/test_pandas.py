import io
import json
import os.path
import posixpath
import tempfile
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterator, List, Type, Union

import pandas as pd
import pytest
from fsspec.implementations.local import LocalFileSystem
from pydantic import parse_obj_as
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from mlem.api.commands import import_object
from mlem.constants import MLEM_CONFIG_FILE_NAME
from mlem.contrib.pandas import (
    PANDAS_FORMATS,
    PANDAS_SERIES_FORMATS,
    DataFrameType,
    PandasConfig,
    PandasFormat,
    PandasReader,
    PandasSeriesReader,
    PandasSeriesWriter,
    PandasWriter,
    SeriesType,
    get_pandas_batch_formats,
    pd_type_from_string,
    python_type_from_pd_string_repr,
    python_type_from_pd_type,
    string_repr_from_pd_type,
)
from mlem.core.artifacts import LOCAL_STORAGE
from mlem.core.data_type import DataAnalyzer, DataReader, DataType, DataWriter
from mlem.core.errors import (
    DeserializationError,
    SerializationError,
    UnsupportedDataBatchLoadingType,
)
from mlem.core.meta_io import MLEM_EXT
from mlem.core.metadata import load, save
from mlem.core.model import Signature
from mlem.core.objects import MlemData
from mlem.utils.module import get_object_requirements
from tests.conftest import data_write_read_check, long

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


def df_to_str(df: Union[pd.DataFrame, pd.Series]):
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
    return DataAnalyzer.analyze(data)


@pytest.fixture
def df_type2(data2):
    return DataAnalyzer.analyze(data2)


@pytest.fixture
def series_data(data):
    res = data.iloc[0]
    res.name = str(res.name)
    return res


@pytest.fixture
def series_df_type(series_data):
    return DataAnalyzer.analyze(series_data)


@pytest.fixture
def series_data2(data2):
    res = data2.iloc[0]
    res.name = str(res.name)
    return res


@pytest.fixture
def series_df_type2(series_data2):
    return DataAnalyzer.analyze(series_data2)


def for_all_formats(
    valid_formats: Dict[str, PandasFormat],
    exclude: Union[List[str], Callable] = None,
):
    ex = exclude if isinstance(exclude, list) else []
    formats = [name for name in valid_formats if name not in ex]
    mark = pytest.mark.parametrize("format", formats)
    if isinstance(exclude, list):
        return mark
    return mark(ex)


def data_write_read_batch_check(
    data_type: DataType,
    format: str,
    reader_type: Type[DataReader] = None,
    custom_eq: Callable[[Any, Any], bool] = None,
):
    with tempfile.TemporaryDirectory() as tmpdir:
        BATCH_SIZE = 2
        storage = LOCAL_STORAGE

        fmt = get_pandas_batch_formats(BATCH_SIZE)[format]
        art = fmt.write(
            data_type.data, storage, posixpath.join(tmpdir, "data")
        )
        reader = PandasReader(data_type=data_type, format=format)
        artifacts = {"data": art}
        if reader_type is not None:
            assert isinstance(reader, reader_type)

        df_iterable: Iterator = reader.read_batch(artifacts, BATCH_SIZE)
        df = None
        col_types = None
        while True:
            try:
                chunk = next(df_iterable)
                if df is None:
                    df = pd.DataFrame(columns=chunk.columns, dtype=col_types)
                    col_types = dict(zip(chunk.columns, chunk.dtypes))
                    df = df.astype(dtype=col_types)
                df = pd.concat([df, chunk.data], ignore_index=True)
            except StopIteration:
                break

        assert custom_eq(df, data_type.data)


def data_write_read_batch_unsupported(
    data_type: DataType,
    batch: int,
    writer: DataWriter = None,
    reader_type: Type[DataReader] = None,
):
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = writer or data_type.get_writer()

        storage = LOCAL_STORAGE
        reader, artifacts = writer.write(
            data_type, storage, posixpath.join(tmpdir, "data")
        )
        if reader_type is not None:
            assert isinstance(reader, reader_type)

        with pytest.raises(
            UnsupportedDataBatchLoadingType,
            match="Batch-loading data of type.*",
        ):
            iter = reader.read_batch(artifacts, batch)
            next(iter)


@for_all_formats(valid_formats=PANDAS_FORMATS)
def test_simple_df(data, format):
    writer = PandasWriter(format=format)
    data_write_read_check(
        DataType.create(data), writer, PandasReader, pd.DataFrame.equals
    )


@for_all_formats(valid_formats=PANDAS_SERIES_FORMATS)
def test_simple_series(series_data, format):
    writer = PandasSeriesWriter(format=format)
    data_write_read_check(
        DataType.create(series_data),
        writer,
        PandasSeriesReader,
        pd.Series.equals,
    )


@pytest.mark.parametrize("format", ["csv", "json", "stata"])
def test_simple_batch_df(data, format):
    data_write_read_batch_check(
        DataType.create(data),
        format,
        PandasReader,
        pd.DataFrame.equals,
    )


@for_all_formats(
    valid_formats=PANDAS_FORMATS,
    exclude=[
        "csv",
        "json",
        "stata",
    ],
)
def test_unsupported_batch_df(data, format):
    writer = PandasWriter(format=format)
    data_write_read_batch_unsupported(
        DataType.create(data), 2, writer, PandasReader
    )


@for_all_formats(valid_formats=PANDAS_FORMATS)
def test_with_index(data, format):
    writer = PandasWriter(format=format)
    data_write_read_check(
        DataType.create(data.set_index("a")),
        writer,
        PandasReader,
        custom_assert=pandas_assert,
    )


@for_all_formats(valid_formats=PANDAS_SERIES_FORMATS)
def test_series_with_multiindex(format):
    data = pd.Series([1, 3, 5], index=[["a", "a", "b"], ["1", "2", "2"]])
    writer = PandasSeriesWriter(format=format)
    data_write_read_check(
        DataType.create(data),
        writer,
        PandasSeriesReader,
        custom_assert=pd.Series.equals,
    )


@for_all_formats(valid_formats=PANDAS_FORMATS)
def test_with_multiindex(data, format):
    writer = PandasWriter(format=format)
    data_write_read_check(
        DataType.create(data.set_index(["a", "b"])),
        writer,
        PandasReader,
        custom_assert=pandas_assert,
    )


@for_all_formats(
    valid_formats=PANDAS_FORMATS,
    exclude=[
        "excel",  # Excel does not support datetimes with timezones
        "parquet",  # Casting from timestamp[ns] to timestamp[ms] would lose data
        "stata",  # Data type datetime64[ns, UTC] not supported.
    ],
)
@pytest.mark.parametrize(
    "data", [PD_DATA_FRAME, PD_DATA_FRAME_INDEX, PD_DATA_FRAME_MULTIINDEX]
)
def test_with_index_complex(data, format):
    writer = PandasWriter(format=format)
    data_write_read_check(
        DataType.create(data),
        writer,
        PandasReader,
        custom_assert=pandas_assert,
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

    obj = df_type.dict()
    new_df_type = parse_obj_as(DataType, obj)

    assert df_type == new_df_type


@pytest.mark.parametrize(
    "series_df_type_fx", ["series_df_type2", "series_df_type"]
)
def test_series_df_type(series_df_type_fx, request):
    series_df_type = request.getfixturevalue(series_df_type_fx)
    assert isinstance(series_df_type, SeriesType)

    obj = series_df_type.dict()
    new_df_type = parse_obj_as(DataType, obj)

    assert series_df_type == new_df_type


def test_dataframe_type(df_type: DataFrameType, data):
    assert df_type.get_requirements().modules == ["pandas"]

    obj = df_type.serialize(data)
    payload = json.dumps(obj)
    loaded = json.loads(payload)
    data2 = df_type.deserialize(loaded)

    assert data.equals(data2)


def test_series_type(series_df_type: SeriesType, series_data):
    assert series_df_type.get_requirements().modules == ["pandas"]

    obj = series_df_type.serialize(series_data)
    payload = json.dumps(obj)
    loaded = json.loads(payload)
    data2 = series_df_type.deserialize(loaded)

    assert series_data.equals(data2)


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
    df_type = DataAnalyzer.analyze(data)
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
    df_type: DataFrameType = DataAnalyzer.analyze(df)

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
    df_type: DataFrameType = DataAnalyzer.analyze(df)

    obj = df_type.serialize(df)
    payload = json.dumps(obj)
    loaded = json.loads(payload)
    data = df_type.deserialize(loaded)

    assert df is not data
    pandas_assert(data, df)


@pytest.fixture()
def iris_data():
    data, y = load_iris(return_X_y=True, as_frame=True)
    data["target"] = y
    train_data, _ = train_test_split(data, random_state=42)
    return train_data


def test_save_load(iris_data, tmpdir):
    tmpdir = str(tmpdir / "data")
    save(iris_data, tmpdir)
    data2 = load(tmpdir)

    pandas_assert(data2, iris_data)


@pytest.fixture
def write_csv():
    def write(data, path, fs=None):
        fs = fs or LocalFileSystem()
        with fs.open(path, "wb") as f:
            f.write(data)

    return write


def _check_data(meta, out_path, fs=None):
    fs = fs or LocalFileSystem()
    assert isinstance(meta, MlemData)
    dt = meta.data_type
    assert isinstance(dt, DataFrameType)
    assert dt.columns == ["a", "b"]
    assert dt.dtypes == ["int64", "int64"]
    assert fs.isfile(out_path)
    assert fs.isfile(out_path + MLEM_EXT)
    loaded = load(out_path)
    assert isinstance(loaded, pd.DataFrame)


@pytest.mark.parametrize(
    "file_ext, type_, data",
    [
        (".csv", None, b"a,b\n1,2"),
        ("", "pandas[csv]", b"a,b\n1,2"),
        (".json", None, b'[{"a":1,"b":2}]'),
    ],
)
def test_import_data_csv(tmpdir, write_csv, file_ext, type_, data):
    path = str(tmpdir / "mydata" + file_ext)
    write_csv(data, path)
    target_path = str(tmpdir / "mlem_data")
    meta = import_object(path, target=target_path, type_=type_)
    _check_data(meta, target_path)


@long
def test_import_data_csv_remote(s3_tmp_path, s3_storage_fs, write_csv):
    project_path = s3_tmp_path("test_csv_import")
    path = posixpath.join(project_path, "data.csv")
    write_csv(b"a,b\n1,2", path, s3_storage_fs)
    target_path = posixpath.join(project_path, "imported_data")
    meta = import_object(path, target=target_path)
    _check_data(meta, target_path, s3_storage_fs)


def test_default_format(set_mlem_project_root, df_type):
    set_mlem_project_root("pandas", __file__)
    config = PandasConfig()
    assert config.default_format == "json"


def test_dataframe():
    value = pd.DataFrame([{"a": 1}])
    assert DataFrameType.is_object_valid(value)
    dt = DataAnalyzer.analyze(value)
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
    dt2 = parse_obj_as(DataType, payload)
    assert dt2 == dt
    assert dt.get_model().__name__ == "DataFrame"
    assert dt.get_model().schema() == {
        "title": "DataFrame",
        "type": "object",
        "properties": {
            "values": {
                "title": "Values",
                "type": "array",
                "items": {"$ref": "#/definitions/DataFrameRow"},
            }
        },
        "required": ["values"],
        "definitions": {
            "DataFrameRow": {
                "title": "DataFrameRow",
                "type": "object",
                "properties": {"a": {"title": "A", "type": "integer"}},
                "required": ["a"],
            }
        },
    }


def test_infer_format(tmpdir):
    path = str(tmpdir / "mydata.parquet")
    value = pd.DataFrame([{"a": 1}])
    meta = save(value, path)
    assert isinstance(meta, MlemData)
    assert isinstance(meta.reader, PandasReader)
    assert meta.reader.format == "parquet"


def test_series(series_data2: pd.Series, series_df_type2, df_type2):
    assert isinstance(series_df_type2, SeriesType)
    assert all(df_type2.columns == series_data2.index)

    obj = series_df_type2.serialize(series_data2)
    payload = json.dumps(obj)
    loaded = json.loads(payload)
    data = series_df_type2.deserialize(loaded)

    assert isinstance(data, pd.Series)
    assert data is not series_data2
    assert list(data.index) == list(series_data2.index), "different index"
    assert data.shape == series_data2.shape, "different shapes"
    assert df_to_str(data) == df_to_str(
        series_data2
    ), "different str representation"


def test_change_format(mlem_project, data):
    with open(
        os.path.join(mlem_project, MLEM_CONFIG_FILE_NAME),
        "w",
        encoding="utf8",
    ) as f:
        f.write("pandas:\n  default_format: parquet")
    meta = save(data, "data", project=mlem_project)
    assert isinstance(meta, MlemData)
    assert isinstance(meta.data_type, DataFrameType)
    writer = meta.data_type.get_writer(project=mlem_project)
    assert isinstance(writer, PandasWriter)
    assert writer.format == "parquet"
    assert isinstance(meta.reader, PandasReader)
    assert meta.reader.format == "parquet"


def test_signature_req(data):
    def f(x):
        return x

    sig = Signature.from_method(f, auto_infer=True, x=data)

    assert set(get_object_requirements(sig).modules) == {"pandas"}


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
