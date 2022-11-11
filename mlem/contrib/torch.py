"""Torch models support
Extension type: model

ModelType and ModelIO implementations for `torch.nn.Module`
ImportHook for importing files saved with `torch.save`
DataType, Reader and Writer implementations for `torch.Tensor`
"""
import logging
from typing import Any, ClassVar, Iterator, List, Optional, Tuple

import cloudpickle
import torch
from pydantic import conlist, create_model

from mlem.contrib.numpy import python_type_from_np_string_repr
from mlem.core.artifacts import Artifacts, FSSpecArtifact, Storage
from mlem.core.data_type import (
    DataHook,
    DataReader,
    DataSerializer,
    DataType,
    DataWriter,
)
from mlem.core.errors import DeserializationError, SerializationError
from mlem.core.hooks import IsInstanceHookMixin
from mlem.core.import_objects import LoadAndAnalyzeImportHook
from mlem.core.meta_io import Location
from mlem.core.model import ModelHook, ModelIO, ModelType, Signature
from mlem.core.objects import MlemModel
from mlem.core.requirements import InstallableRequirement, Requirements


def python_type_from_torch_string_repr(dtype: str):
    #  not sure this will work all the time
    return python_type_from_np_string_repr(dtype)


logger = logging.getLogger(__name__)


class TorchTensorDataType(
    DataType, DataSerializer, DataHook, IsInstanceHookMixin
):
    """DataType implementation for `torch.Tensor`"""

    type: ClassVar[str] = "torch"
    valid_types: ClassVar = (torch.Tensor,)
    shape: Tuple[Optional[int], ...]
    """Shape of `torch.Tensor` object"""
    dtype: str
    """Type name of `torch.Tensor` elements"""

    def _check_shape(self, tensor, exc_type):
        if tuple(tensor.shape)[1:] != self.shape[1:]:
            raise exc_type(
                f"given tensor is of shape: {(None,) + tuple(tensor.shape)[1:]}, expected: {self.shape}"
            )

    def serialize(self, instance: torch.Tensor):
        self.check_type(instance, torch.Tensor, SerializationError)
        if instance.dtype is not getattr(torch, self.dtype):
            raise SerializationError(
                f"given tensor is of dtype: {instance.dtype}, "
                f"expected: {getattr(torch, self.dtype)}"
            )
        self._check_shape(instance, SerializationError)
        return instance.tolist()

    def deserialize(self, obj):
        try:
            ret = torch.tensor(obj, dtype=getattr(torch, self.dtype))
        except (ValueError, TypeError):
            raise DeserializationError(  # pylint: disable=W0707
                f"given object: {obj} could not be converted to tensor "
                f"of type: {getattr(torch, self.dtype)}"
            )
        self._check_shape(ret, DeserializationError)
        return ret

    def get_requirements(self) -> Requirements:
        return Requirements.new([InstallableRequirement.from_module(torch)])

    def get_writer(
        self, project: str = None, filename: str = None, **kwargs
    ) -> DataWriter:
        return TorchTensorWriter(**kwargs)

    def _subtype(self, subshape: Tuple[Optional[int], ...]):
        if len(subshape) == 0:
            return python_type_from_torch_string_repr(self.dtype)
        return conlist(
            self._subtype(subshape[1:]),
            min_items=subshape[0],
            max_items=subshape[0],
        )

    def get_model(self, prefix: str = ""):
        return create_model(
            prefix + "TorchTensor",
            __root__=(List[self._subtype(self.shape[1:])], ...),  # type: ignore
        )

    @classmethod
    def process(cls, obj: torch.Tensor, **kwargs) -> DataType:
        return TorchTensorDataType(
            shape=(None,) + obj.shape[1:],
            dtype=str(obj.dtype)[len("torch") + 1 :],
        )


class TorchTensorWriter(DataWriter):
    """Write torch tensors"""

    type: ClassVar[str] = "torch"

    def write(
        self, data: DataType, storage: Storage, path: str
    ) -> Tuple[DataReader, Artifacts]:
        with storage.open(path) as (f, art):
            torch.save(data.data, f)
        return TorchTensorReader(data_type=data), {self.art_name: art}


class TorchTensorReader(DataReader):
    """Read torch tensors"""

    type: ClassVar[str] = "torch"

    def read(self, artifacts: Artifacts) -> DataType:
        if DataWriter.art_name not in artifacts:
            raise ValueError(
                f"Wrong artifacts {artifacts}: should be one {DataWriter.art_name} file"
            )
        with artifacts[DataWriter.art_name].open() as f:
            data = torch.load(f)
            return self.data_type.copy().bind(data)

    def read_batch(
        self, artifacts: Artifacts, batch_size: int
    ) -> Iterator[DataType]:
        raise NotImplementedError


class TorchModelIO(ModelIO):
    """IO for PyTorch models"""

    type: ClassVar[str] = "torch_io"
    is_jit: bool = False
    """Is model jit compiled"""

    def dump(self, storage: Storage, path, model) -> Artifacts:
        self.is_jit = isinstance(model, torch.jit.ScriptModule)
        with storage.open(path) as (f, art):
            if self.is_jit:
                torch.jit.save(model, f)
            else:
                torch.save(model, f, pickle_module=cloudpickle)
            return {self.art_name: art}

    def load(self, artifacts: Artifacts):
        if len(artifacts) != 1:
            raise ValueError("Invalid artifacts: should have only one file")

        load = torch.jit.load if self.is_jit else torch.load
        with artifacts[self.art_name].open() as f:
            return load(f)


class TorchModel(ModelType, ModelHook, IsInstanceHookMixin):
    """
    :class:`.ModelType` implementation for PyTorch models
    """

    type: ClassVar[str] = "torch"
    valid_types: ClassVar = (torch.nn.Module,)
    io: ModelIO = TorchModelIO()
    """TorchModelIO"""

    @classmethod
    def process(
        cls, obj: Any, sample_data: Optional[Any] = None, **kwargs
    ) -> ModelType:
        signature = Signature.from_method(
            obj.__call__,
            sample_data,
            override_name="__call__",
            auto_infer=sample_data is not None,
        )
        return TorchModel(
            model=obj,
            methods={
                "__call__": signature,
            },
        )

    def __call__(self, data):
        if isinstance(data, torch.Tensor):
            return self.model(data)
        return self.model(*data)

    def get_requirements(self) -> Requirements:
        return super().get_requirements() + InstallableRequirement.from_module(
            mod=torch
        )


class TorchModelImport(LoadAndAnalyzeImportHook):
    """Import torch models saved with `torch.save`"""

    type: ClassVar = "torch"
    force_type: ClassVar = MlemModel

    @classmethod
    def is_object_valid(cls, obj: Location) -> bool:
        # TODO only manual import type specification for now
        return False

    @classmethod
    def load_obj(cls, location: Location, modifier: Optional[str], **kwargs):
        torch_obj = TorchModelIO().load(
            {
                TorchModelIO.art_name: FSSpecArtifact(
                    uri=location.uri, size=0, hash=""
                )
            }
        )
        logger.debug("Loaded dataframe object\n %s", torch_obj)
        return torch_obj


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
