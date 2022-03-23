import posixpath
import re
from abc import ABC
from dataclasses import dataclass
from typing import (
    IO,
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
from pandas import Int64Dtype, SparseDtype, StringDtype
from pandas.core.dtypes.dtypes import (
    CategoricalDtype,
    DatetimeTZDtype,
    IntervalDtype,
    PandasExtensionDtype,
    PeriodDtype,
)
from pydantic import BaseModel, create_model, validator

from mlem.config import MlemConfigBase
from mlem.contrib.numpy import np_type_from_string, python_type_from_np_type
from mlem.core.artifacts import (
    Artifact,
    Artifacts,
    PlaceholderArtifact,
    Storage,
    get_file_info,
)
from mlem.core.dataset_type import (
    Dataset,
    DatasetHook,
    DatasetReader,
    DatasetSerializer,
    DatasetType,
    DatasetWriter,
)
from mlem.core.errors import DeserializationError, SerializationError
from mlem.core.import_objects import ExtImportHook
from mlem.core.meta_io import Location
from mlem.core.metadata import get_object_metadata
from mlem.core.objects import MlemMeta
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
    except ValueError as e:
        for dtype, pattern in PD_EXT_TYPES.items():
            if pattern.match(string_repr) is not None:
                return dtype.construct_from_string(string_repr)
        raise ValueError(f"unknown pandas dtype {string_repr}") from e


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


class PandasConfig(MlemConfigBase):
    DEFAULT_FORMAT: str = "csv"

    class Config:
        section = "pandas"


PANDAS_CONFIG = PandasConfig()


class _PandasDatasetType(
    LibRequirementsMixin, DatasetType, DatasetHook, DatasetSerializer, ABC
):
    """Intermidiate class for pandas DatasetType implementations

    :param columns: list of column names (including index)
    :param dtypes: list of string representations of pandas dtypes of columns
    :param index_cols: list of column names that are used as index"""

    libraries: ClassVar = [pd]
    columns: List[str]
    dtypes: List[str]
    index_cols: List[str]

    @classmethod
    def process(cls, obj: Any, **kwargs) -> "_PandasDatasetType":
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

    def deserialize(self, obj):
        self.check_type(obj, dict, DeserializationError)
        try:
            ret = pd.DataFrame.from_records(obj["values"])
        except (ValueError, KeyError) as e:
            raise DeserializationError(
                f"given object: {obj} could not be converted to dataframe"
            ) from e

        self._validate_columns(
            ret, DeserializationError
        )  # including index columns
        ret = self.align_types(ret)  # including index columns
        self._validate_dtypes(ret, DeserializationError)
        return self.align_index(ret)

    def serialize(self, instance: pd.DataFrame):
        self.check_type(instance, pd.DataFrame, SerializationError)
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


class SeriesType(_PandasDatasetType):
    """
    :class:`.DatasetType` implementation for `pandas.Series` objects which stores them as built-in Python dicts

    """

    type: ClassVar[str] = "series"

    def get_model(self):
        return create_model(
            "Series",
            **{
                c: (python_type_from_pd_string_repr(t), ...)
                for c, t in zip(self.columns, self.dtypes)
            },
        )

    @classmethod
    def is_object_valid(cls, obj: Any) -> bool:
        return isinstance(obj, pd.Series)

    @classmethod
    def process(cls, obj: pd.Series, **kwargs) -> "_PandasDatasetType":
        return super().process(
            pd.DataFrame([obj], index=None).reset_index(drop=True)
        )

    def deserialize(self, obj):
        return super().deserialize({"values": [obj]}).iloc[0]

    def serialize(self, instance: pd.Series):
        return super().serialize(
            pd.DataFrame([instance]).reset_index(drop=True)
        )["values"][0]

    def get_writer(self, **kwargs) -> "DatasetWriter":
        raise NotImplementedError


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


class DataFrameType(_PandasDatasetType):
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

    def align(self, df):
        return self.align_index(self.align_types(df))

    def row_type(self):
        return create_model(
            "DataFrameRow",
            **{
                c: (python_type_from_pd_string_repr(t), ...)
                for c, t in zip(self.columns, self.dtypes)
            },
        )

    def get_writer(self, **kwargs):
        fmt = kwargs.get("format", PANDAS_CONFIG.DEFAULT_FORMAT)
        return PandasWriter(format=fmt)


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


class ToBytesIO(IO[Any]):
    def __init__(self, fileobj, encoding: str = "utf8"):
        self.encoding = encoding
        self.fileobj = fileobj

    def write(self, payload: str):
        self.fileobj.write(payload.encode(self.encoding))


@dataclass
class PandasFormat:
    read_func: Callable
    write_func: Callable
    read_args: Optional[Dict[str, Any]] = None
    write_args: Optional[Dict[str, Any]] = None
    file_name: str = "data.pd"
    string_buffer: bool = False

    def read(self, artifacts: Artifacts, **kwargs):
        """Read DataFrame"""
        read_kwargs = {}
        if self.read_args:
            read_kwargs.update(self.read_args)
        read_kwargs.update(kwargs)
        if len(artifacts) != 1:
            raise ValueError(
                f"Wrong artifacts {artifacts}: should be one {self.file_name} file"
            )
        with artifacts[0].open() as f:
            return self.read_func(f, **read_kwargs)

    def write(
        self, df: pd.DataFrame, storage: Storage, path: str, **kwargs
    ) -> Artifact:
        write_kwargs = {}
        if self.write_args:
            write_kwargs.update(self.write_args)
        write_kwargs.update(kwargs)

        if has_index(df):
            df = reset_index(df)

        with storage.open(path) as (f, art):
            if self.string_buffer:
                f = ToBytesIO(f)  # type: ignore[abstract]
            self.write_func(df, f, **write_kwargs)
            return art


def read_csv_with_unnamed(*args, **kwargs):
    df = pd.read_csv(*args, **kwargs)
    unnamed = {}
    for col in df.columns:
        if col.startswith("Unnamed: "):
            unnamed[col] = ""
    if not unnamed:
        return df
    return df.rename(unnamed, axis=1)  # pylint: disable=no-member


def read_json_reset_index(*args, **kwargs):
    return pd.read_json(*args, **kwargs).reset_index(drop=True)


def read_html(*args, **kwargs):
    # read_html returns list of dataframes
    return pd.read_html(*args, **kwargs)[0]


PANDAS_FORMATS = {
    "csv": PandasFormat(
        read_csv_with_unnamed,
        pd.DataFrame.to_csv,
        file_name="data.csv",
        write_args={"index": False},
    ),
    "json": PandasFormat(
        read_json_reset_index,
        pd.DataFrame.to_json,
        file_name="data.json",
        write_args={"date_format": "iso", "date_unit": "ns"},
    ),
    "html": PandasFormat(
        read_html,
        pd.DataFrame.to_html,
        file_name="data.html",
        write_args={"index": False},
        string_buffer=True,
    ),
    "excel": PandasFormat(
        pd.read_excel,
        pd.DataFrame.to_excel,
        file_name="data.xlsx",
        write_args={"index": False},
    ),
    "parquet": PandasFormat(
        pd.read_parquet, pd.DataFrame.to_parquet, file_name="data.parquet"
    ),
    "feather": PandasFormat(
        pd.read_feather, pd.DataFrame.to_feather, file_name="data.feather"
    ),
    "pickle": PandasFormat(  # TODO buffer closed error for some reason
        pd.read_pickle, pd.DataFrame.to_pickle, file_name="data.pkl"
    ),
    "strata": PandasFormat(  # TODO int32 converts to int64 for some reason
        pd.read_stata, pd.DataFrame.to_stata, write_args={"write_index": False}
    ),
}


class _PandasIO(BaseModel):
    format: str

    @validator("format")
    def is_valid_format(  # pylint: disable=no-self-argument
        cls, value  # noqa: B902
    ):
        if value not in PANDAS_FORMATS:
            raise ValueError(f"format {value} is not supported")
        return value

    @property
    def fmt(self):
        return PANDAS_FORMATS[self.format]


class PandasReader(_PandasIO, DatasetReader):
    """DatasetReader for pandas dataframes"""

    type: ClassVar[str] = "pandas"
    dataset_type: DataFrameType

    def read(self, artifacts: Artifacts) -> Dataset:
        return Dataset(
            self.dataset_type.align(self.fmt.read(artifacts)),
            self.dataset_type,
        )


class PandasWriter(DatasetWriter, _PandasIO):
    """DatasetWriter for pandas dataframes"""

    type: ClassVar[str] = "pandas"

    def write(
        self, dataset: Dataset, storage: Storage, path: str
    ) -> Tuple[DatasetReader, Artifacts]:
        fmt = self.fmt
        art = fmt.write(dataset.data, storage, path)
        if not isinstance(dataset.dataset_type, DataFrameType):
            raise ValueError("Cannot write non-pandas Dataset")
        return PandasReader(
            dataset_type=dataset.dataset_type, format=self.format
        ), [art]


class PandasImport(ExtImportHook):
    EXTS = tuple(f".{k}" for k in PANDAS_FORMATS)
    type_ = "pandas"

    @classmethod
    def is_object_valid(cls, obj: Location) -> bool:
        return super().is_object_valid(obj) and obj.fs.isfile(obj.fullpath)

    @classmethod
    def process(
        cls,
        obj: Location,
        copy_data: bool = True,
        modifier: Optional[str] = None,
        **kwargs,
    ) -> MlemMeta:
        ext = modifier or posixpath.splitext(obj.path)[1][1:]
        fmt = PANDAS_FORMATS[ext]
        read_args = fmt.read_args or {}
        read_args.update(kwargs)
        with obj.open("rb") as f:
            data = fmt.read_func(f, **read_args)
        meta = get_object_metadata(data)
        if not copy_data:
            meta.artifacts = [
                PlaceholderArtifact(
                    location=obj,
                    uri=obj.uri,
                    **get_file_info(obj.fullpath, obj.fs),
                )
            ]
        return meta
