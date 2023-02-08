"""Scipy Sparse matrices support
Extension type: data

DataType, Reader and Writer implementations for `scipy.sparse`
"""
from typing import ClassVar, Iterator, List, Optional, Tuple, Type, Union

import scipy
from pydantic import BaseModel
from pydantic.main import create_model
from pydantic.types import conlist
from scipy import sparse
from scipy.sparse import spmatrix

from mlem.contrib.numpy import (
    check_shape,
    np_type_from_string,
    python_type_from_np_string_repr,
)
from mlem.core.artifacts import Artifacts, Storage
from mlem.core.data_type import (
    DataHook,
    DataReader,
    DataSerializer,
    DataType,
    DataWriter,
    WithDefaultSerializer,
)
from mlem.core.errors import DeserializationError, SerializationError
from mlem.core.hooks import IsInstanceHookMixin
from mlem.core.requirements import InstallableRequirement, Requirements


class ScipySparseMatrix(
    WithDefaultSerializer, DataType, DataHook, IsInstanceHookMixin
):
    """
    DataType implementation for scipy sparse matrix
    """

    type: ClassVar[str] = "csr_matrix"
    valid_types: ClassVar = (spmatrix,)
    shape: Tuple[int, ...]
    """Shape of `sparse.csr_matrix` object in data"""
    dtype: str
    """Dtype of `sparse.csr_matrix` object in data"""

    def get_requirements(self) -> Requirements:
        return Requirements.new([InstallableRequirement.from_module(scipy)])

    @classmethod
    def process(cls, obj: sparse.csr_matrix, **kwargs) -> DataType:
        return ScipySparseMatrix(dtype=obj.dtype.name, shape=obj.shape)

    def get_writer(
        self, project: str = None, filename: str = None, **kwargs
    ) -> DataWriter:
        return ScipyWriter(**kwargs)

    def subtype(self, subshape: Tuple[Optional[int], ...]):
        if len(subshape) == 0:
            return python_type_from_np_string_repr(self.dtype)
        return conlist(
            self.subtype(subshape[1:]),
            min_items=subshape[0],
            max_items=subshape[0],
        )


class ScipyWriter(DataWriter[ScipySparseMatrix]):
    """
    Write scipy matrix to npz format
    """

    type: ClassVar[str] = "csr_matrix"

    def write(
        self, data: DataType, storage: Storage, path: str
    ) -> Tuple[DataReader, Artifacts]:
        with storage.open(path) as (f, art):
            sparse.save_npz(f, data.data)
        return ScipyReader(data_type=data), {self.art_name: art}


class ScipyReader(DataReader):
    """
    Read scipy matrix from npz format
    """

    type: ClassVar[str] = "csr_matrix"

    def read_batch(
        self, artifacts: Artifacts, batch_size: int
    ) -> Iterator[DataType]:
        raise NotImplementedError

    def read(self, artifacts: Artifacts) -> Iterator[DataType]:
        if DataWriter.art_name not in artifacts:
            raise ValueError(
                f"Wrong artifacts {artifacts}: should be one {DataWriter.art_name} file"
            )
        with artifacts[DataWriter.art_name].open() as f:
            data = sparse.load_npz(f)
        return self.data_type.copy().bind(data)


class ScipySparseMatrixSerializer(DataSerializer[ScipySparseMatrix]):
    """
    Serializer for scipy sparse matrices
    """

    is_default: ClassVar = True
    data_class: ClassVar = ScipySparseMatrix

    def get_model(
        self, data_type: ScipySparseMatrix, prefix: str = ""
    ) -> Union[Type[BaseModel], type]:
        subtype = data_type.subtype(data_type.shape[1:])
        item_type = List[subtype]  # type: ignore[valid-type,index]
        return create_model(
            prefix + "ScipySparse",
            __root__=(item_type, ...),
        )

    def serialize(self, data_type: ScipySparseMatrix, instance: spmatrix):
        data_type.check_type(instance, sparse.csr_matrix, SerializationError)
        if instance.dtype != np_type_from_string(data_type.dtype):
            raise SerializationError(
                f"given matrix is of dtype: {instance.dtype}, "
                f"expected: {data_type.dtype}"
            )
        check_shape(data_type.shape, instance, SerializationError)
        coordinate_matrix = instance.tocoo()
        data = coordinate_matrix.data
        row = coordinate_matrix.row
        col = coordinate_matrix.col
        return data, (row, col)

    def deserialize(self, data_type, obj) -> sparse.csr_matrix:
        try:
            mat = sparse.csr_matrix(
                obj, dtype=data_type.dtype, shape=data_type.shape
            )
        except ValueError as e:
            raise DeserializationError(
                f"Given object {obj} could not be converted"
                f"to sparse matrix of type: {data_type.type}"
            ) from e
        check_shape(data_type.shape, mat, DeserializationError)
        return mat
