import os
import re
from abc import ABC
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

import numpy as np
import pandas as pd
from fsspec import AbstractFileSystem
from pandas import Int64Dtype, SparseDtype, StringDtype
from pandas.core.dtypes.dtypes import (
    CategoricalDtype,
    DatetimeTZDtype,
    IntervalDtype,
    PandasExtensionDtype,
    PeriodDtype,
)
from pydantic import BaseModel, create_model, validator

from mlem.contrib.numpy import np_type_from_string, python_type_from_np_type
from mlem.core.artifacts import Artifacts
from mlem.core.dataset_type import (
    Dataset,
    DatasetHook,
    DatasetReader,
    DatasetSerializer,
    DatasetType,
    DatasetWriter,
)
from mlem.core.errors import DeserializationError, SerializationError
from mlem.core.requirements import LibRequirementsMixin

_PD_EXT_TYPES = {
    DatetimeTZDtype: r"datetime64.*",
    CategoricalDtype: r"category",
    PeriodDtype: r"[pP]eriod.*",
    SparseDtype: r"Sparse.*",
    IntervalDtype: r"[iI]nterval.*",
    Int64Dtype: r"U?Int\d*",
    StringDtype: r"string",
}
PD_EXT_TYPES = {
    dtype: re.compile(pattern) for dtype, pattern in _PD_EXT_TYPES.items()
}


def string_repr_from_pd_type(
    dtype: Union[np.dtype, PandasExtensionDtype]
) -> str:
    """Returns string representation of pandas dtype"""
    return dtype.name


def pd_type_from_string(string_repr):
    """Creates pandas dtype from string representation"""
    try:
        return np_type_from_string(string_repr)
    except ValueError:
        for dtype, pattern in PD_EXT_TYPES.items():
            if pattern.match(string_repr) is not None:
                return dtype.construct_from_string(string_repr)
        raise ValueError(f"unknown pandas dtype {string_repr}")


def python_type_from_pd_type(pd_type: Union[np.dtype, Type]):
    """Returns python builtin type that corresponds to pandas dtype"""
    if isinstance(
        pd_type, np.dtype
    ):  # or (isinstance(pd_type, type) and pd_type.__module__ == 'numpy'):
        return python_type_from_np_type(pd_type)
    return str


def python_type_from_pd_string_repr(string_repr: str) -> type:
    """Returns python type from pandas dtype string representation"""
    pd_type = pd_type_from_string(string_repr)
    return python_type_from_pd_type(pd_type)


def need_string_value(dtype):
    """Returns true if dtype must be cast to str for serialization"""
    return not isinstance(dtype, np.dtype) or dtype.type in (
        np.datetime64,
        np.object_,
    )


class _PandasDatasetType(LibRequirementsMixin, DatasetType, DatasetHook, ABC):
    """Intermidiate class for pandas DatasetType implementations

    :param columns: list of column names (including index)
    :param dtypes: list of string representations of pandas dtypes of columns
    :param index_cols: list of column names that are used as index"""

    libraries: ClassVar = [pd]
    columns: List[str]
    dtypes: List[str]
    index_cols: List[str]

    @classmethod
    def process(cls, obj: Any, **kwargs) -> DatasetType:
        if has_index(obj):
            index_cols, obj = _reset_index(obj)
        else:
            index_cols = []

        return cls(
            columns=list(obj.columns),
            dtypes=[string_repr_from_pd_type(d) for d in obj.dtypes],
            index_cols=index_cols,
        )

    @property
    def actual_dtypes(self):
        """List of pandas dtypes for columns"""
        return [pd_type_from_string(s) for s in self.dtypes]

    def _validate_columns(self, df: pd.DataFrame, exc_type):
        """Validates that df has correct columns"""
        if set(df.columns) != set(self.columns):
            raise exc_type(
                f"given dataframe has columns: {list(df.columns)}, expected: {self.columns}"
            )

    def _validate_dtypes(self, df: pd.DataFrame, exc_type):
        """Validates that df has correct column dtypes"""
        df = df[self.columns]
        for col, expected, dtype in zip(self.columns, self.dtypes, df.dtypes):
            pd_type = string_repr_from_pd_type(dtype)
            if expected != pd_type:
                raise exc_type(
                    f"given dataframe has incorrect type {pd_type} in column {col}. Expected: {expected}"
                )


class SeriesType(_PandasDatasetType, DatasetHook):
    """
    :class:`.DatasetType` implementation for `pandas.Series` objects which stores them as built-in Python dicts

    """

    type: ClassVar = "series"

    @classmethod
    def is_object_valid(cls, obj: Any) -> bool:
        return False  # isinstance(obj, pd.Series) TODO https://github.com/iterative/mlem/issues/32

    def deserialize(self, obj):
        return pd.Series(obj)

    def serialize(self, instance: pd.Series):
        return instance.to_dict()

    # def get_spec(self):
    #     return [Field(c, python_type_from_pd_string_repr(d), False) for c, d in zip(self.columns, self.dtypes)]


def has_index(df: pd.DataFrame):
    """Returns true if df has non-trivial index"""
    return not isinstance(df.index, pd.RangeIndex)


def _reset_index(df: pd.DataFrame):
    """Transforms indexes to columns"""
    index_name = df.index.name or ""  # save it for future renaming
    cols = set(df.columns)
    df = df.reset_index()  # can rename indexes if they didnt have a name
    index_cols = [
        c for c in df.columns if c not in cols
    ]  # get new columns - they were index columns
    if len(index_cols) == 1:
        df = df.rename({index_cols[0]: index_name}, axis=1)
        index_cols = [index_name]
    return index_cols, df


def reset_index(df: pd.DataFrame, return_copied=False):
    """Transforms indexes to columns if index is non-trivial"""
    if has_index(df):
        _, df = _reset_index(df)
        if return_copied:
            return True, df
        return df
    if return_copied:
        return False, df
    return df


class DataFrameType(_PandasDatasetType, DatasetSerializer):
    """
    :class:`.DatasetType` implementation for `pandas.DataFrame`
    """

    type: ClassVar[str] = "dataframe"

    @classmethod
    def is_object_valid(cls, obj: Any) -> bool:
        return isinstance(obj, pd.DataFrame)

    def get_model(self) -> Type[BaseModel]:
        # TODO: https://github.com/iterative/mlem/issues/33
        return create_model("DataFrame", values=(List[self.row_type()], ...))  # type: ignore

    def deserialize(self, obj):
        self._check_type(obj, dict, DeserializationError)
        try:
            ret = pd.DataFrame.from_records(obj["values"])
        except (ValueError, KeyError):
            raise DeserializationError(
                f"given object: {obj} could not be converted to dataframe"
            )

        self._validate_columns(
            ret, DeserializationError
        )  # including index columns
        ret = self.align_types(ret)  # including index columns
        self._validate_dtypes(ret, DeserializationError)
        return self.align_index(ret)

    def align_types(self, df):
        """Restores column order and casts columns to expected types"""
        df = df[self.columns]
        for col, expected, dtype in zip(
            self.columns, self.actual_dtypes, df.dtypes
        ):
            if expected != dtype:
                df[col] = df[col].astype(expected)
        return df

    def align_index(self, df):
        """Transform index columns to actual indexes"""
        if len(self.index_cols) > 0:
            df = df.set_index(self.index_cols)
        return df

    def align(self, df):
        return self.align_index(self.align_types(df))

    def serialize(self, instance: pd.DataFrame):
        self._check_type(instance, pd.DataFrame, SerializationError)
        is_copied, instance = reset_index(instance, return_copied=True)

        self._validate_columns(instance, SerializationError)
        self._validate_dtypes(instance, SerializationError)

        for col, dtype in zip(self.columns, self.actual_dtypes):
            if need_string_value(dtype):
                if not is_copied:
                    instance = instance.copy()
                    is_copied = True
                instance[col] = instance[col].astype("string")

        return {"values": (instance.to_dict("records"))}

    # def get_spec(self) -> ArgList:
    #     return [Field('values', List[self.row_type], False)]

    def row_type(self):
        # return SeriesType(columns=self.columns, dtypes=self.dtypes, index_cols=self.index_cols)
        return create_model(
            "DataFrameRow",
            **{
                c: (python_type_from_pd_string_repr(t), ...)
                for c, t in zip(self.columns, self.dtypes)
            },
        )

    def get_writer(self, **kwargs):
        format = kwargs.get("format", "csv")
        return PandasWriter(format=format)  # TODO env configuration


# class PandasFormat2:
#     """ABC for reading and writing different formats supported in pandas
#
#     :param read_args: additional arguments for reading
#     :param write_args: additional arguments for writing
#     """
#     # renamed type to _type so mypy and flake8 won't complain
#     _type: str = None
#     read_func: typing.Callable = None
#     write_func: typing.Callable = None
#     buffer_type: typing.Type[typing.IO] = None
#
#     def __init__(self, read_args: Dict[str, Any] = None, write_args: Dict[str, Any] = None):
#         self.write_args = write_args or {}
#         self.read_args = read_args or {}
#
#     def read(self, file_or_path):
#         """Read DataFrame
#
#         :param file_or_path: source for read function"""
#         kwargs = self.add_read_args()
#         kwargs.update(self.read_args)
#         return type(self).read_func(file_or_path, **kwargs)
#
#     def write(self, dataframe) -> typing.IO:
#         """Write DataFrame to buffer
#
#         :param dataframe: DataFrame to write
#         """
#         buf = self.buffer_type()
#         kwargs = self.add_write_args()
#         kwargs.update(self.write_args)
#         if has_index(dataframe):
#             dataframe = reset_index(dataframe)
#         type(self).write_func(dataframe, buf, **kwargs)
#         return buf
#
#     def add_read_args(self) -> Dict[str, Any]:
#         """Fuction with additional read argumnets for child classes to override"""
#         return {}
#
#     def add_write_args(self) -> Dict[str, Any]:
#         """Fuction with additional write argumnets for child classes to override"""
#         return {}


#
# class PandasFormatJson(PandasFormat):
#     type = 'json'
#     read_func = pd.read_json
#     write_func = pd.DataFrame.to_json
#     buffer_type = io.StringIO
#
#     def add_write_args(self) -> Dict[str, Any]:
#         return {'date_format': 'iso', 'date_unit': 'ns'}
#
#     def read(self, file_or_path):
#         # read_json creates index for some reason
#         return super(PandasFormatJson, self).read(file_or_path).reset_index(drop=True)
#
#
# class PandasFormatHtml(PandasFormat):
#     type = 'html'
#     read_func = pd.read_html
#     write_func = pd.DataFrame.to_html
#     buffer_type = io.StringIO
#
#     def add_write_args(self) -> Dict[str, Any]:
#         return {'index': False}
#
#     def read(self, file_or_path):
#         # read_html returns list of dataframes
#         df = super(PandasFormatHtml, self).read(file_or_path)
#         return df[0]
#
#
# class PandasFormatExcel(PandasFormat):
#     type = 'excel'
#     read_func = pd.read_excel
#     write_func = pd.DataFrame.to_excel
#     buffer_type = io.BytesIO
#
#     def add_write_args(self) -> Dict[str, Any]:
#         return {'index': False}
#
#
# class PandasFormatHdf(PandasFormat):
#     type = 'hdf'
#     read_func = pd.read_hdf
#     write_func = pd.DataFrame.to_hdf
#     buffer_type = io.BytesIO
#
#     key = 'data'
#
#     def add_write_args(self) -> Dict[str, Any]:
#         return {'key': self.key}
#
#     def write(self, dataframe) -> typing.IO:
#         # to_hdf can write only to file or HDFStore, so there's that
#         kwargs = self.add_write_args()
#         kwargs.update(self.write_args)
#         if has_index(dataframe):
#             dataframe = reset_index(dataframe)
#         path = tempfile.mktemp(suffix='.hd5', dir='.')  # tempfile.TemporaryDirectory breaks on windows for some reason
#         try:
#             type(self).write_func(dataframe, path, **kwargs)
#             with open(path, 'rb') as f:
#                 return self.buffer_type(f.read())
#         finally:
#             os.unlink(path)
#
#     def add_read_args(self) -> Dict[str, Any]:
#         return {'key': self.key}
#
#     def read(self, file_or_path):
#         if not isinstance(file_or_path, str):
#             path = tempfile.mktemp('.hd5', dir='.')
#             try:
#                 with open(path, 'wb') as f:
#                     f.write(file_or_path.read())
#                 df = super().read(path)
#             finally:
#                 os.unlink(path)
#         else:
#             df = super().read(file_or_path)
#
#         return df.reset_index(drop=True)
#
#
# class PandasFormatFeather(PandasFormat):
#     type = 'feather'
#     read_func = pd.read_feather
#     write_func = pd.DataFrame.to_feather
#     buffer_type = io.BytesIO
#
#
# class PandasFormatParquet(PandasFormat):
#     type = 'parquet'
#     read_func = pd.read_parquet
#     write_func = pd.DataFrame.to_parquet
#     buffer_type = io.BytesIO


# class PandasFormatStata(PandasFormat): # TODO int32 converts to int64 for some reason
#     type = 'stata'
#     read_func = pd.read_stata
#     write_func = pd.DataFrame.to_stata
#     buffer_type = io.BytesIO
#
#     def add_write_args(self) -> Dict[str, Any]:
#         return {'write_index': False}
#
# class PandasFormatPickle(PandasFormat): # TODO buffer closed error for some reason
#     type = 'pickle'
#     read_func = pd.read_pickle
#     write_func = pd.DataFrame.to_pickle
#     buffer_type = io.BytesIO


@dataclass
class PandasFormat:
    read_func: Callable
    write_func: Callable
    read_args: Optional[Dict[str, Any]] = None
    write_args: Optional[Dict[str, Any]] = None
    file_name: str = "data.pd"

    def read(self, fs: AbstractFileSystem, path: str, **kwargs):
        """Read DataFrame"""
        read_kwargs = {}
        if self.read_args:
            read_kwargs.update(self.read_args)
        read_kwargs.update(kwargs)

        with fs.open(os.path.join(path, self.file_name), "rb") as f:
            return self.read_func(f, **read_kwargs)

    def write(
        self, df: pd.DataFrame, fs: AbstractFileSystem, path: str, **kwargs
    ):
        write_kwargs = {}
        if self.write_args:
            write_kwargs.update(self.write_args)
        write_kwargs.update(kwargs)

        with fs.open(os.path.join(path, self.file_name), "wb") as f:
            self.write_func(df, f, **write_kwargs)


PANDAS_FORMATS = {
    "csv": PandasFormat(pd.read_csv, pd.DataFrame.to_csv, file_name="data.csv")
}


class _PandasIO(BaseModel):
    format: str

    @validator("format")
    def is_valid_format(cls, value):  # noqa: B902
        if value not in PANDAS_FORMATS:
            raise ValueError(f"format {value} is not supported")
        return value

    @property
    def fmt(self):
        return PANDAS_FORMATS[self.format]


class PandasReader(_PandasIO, DatasetReader):
    """DatasetReader for pandas dataframes"""

    type: ClassVar = "pandas"
    dataset_type: DataFrameType

    def read(self, fs: AbstractFileSystem, path: str) -> Dataset:
        return Dataset(
            self.dataset_type.align(self.fmt.read(fs, path)), self.dataset_type
        )


class PandasWriter(DatasetWriter, _PandasIO):
    """DatasetWriter for pandas dataframes"""

    type: ClassVar = "pandas"

    def write(
        self, dataset: Dataset, fs: AbstractFileSystem, path: str
    ) -> Tuple[DatasetReader, Artifacts]:
        fmt = self.fmt
        fs.makedirs(path, True)
        fmt.write(dataset.data, fs, path)
        if not isinstance(dataset.dataset_type, DataFrameType):
            raise ValueError("Cannot write non-pandas Dataset")
        return PandasReader(
            dataset_type=dataset.dataset_type, format=self.format
        ), [fmt.file_name]
