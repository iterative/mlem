from typing import Any, ClassVar, Optional, Tuple

import torch

from mlem.constants import PREDICT_METHOD_NAME
from mlem.core.artifacts import Artifacts, Storage
from mlem.core.dataset_type import (
    DatasetHook,
    DatasetSerializer,
    DatasetType,
    DatasetWriter,
)
from mlem.core.errors import DeserializationError, SerializationError
from mlem.core.hooks import IsInstanceHookMixin
from mlem.core.model import ModelHook, ModelIO, ModelType, Signature
from mlem.core.requirements import InstallableRequirement, Requirements


class TorchTensorDatasetType(
    DatasetType, DatasetSerializer, DatasetHook, IsInstanceHookMixin
):
    """
    :class:`.DatasetType` implementation for `torch.Tensor` objects
    which converts them to built-in Python lists and vice versa.

    :param shape: shape of `torch.Tensor` objects in dataset
    :param dtype: data type of `torch.Tensor` objects in dataset
    """

    type: ClassVar[str] = "torch"
    valid_types: ClassVar = (torch.Tensor,)
    shape: Tuple[Optional[int], ...]
    dtype: str

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

    def get_writer(self, **kwargs) -> DatasetWriter:
        raise NotImplementedError()

    def get_model(self):
        raise NotImplementedError()

    @classmethod
    def process(cls, obj: torch.Tensor, **kwargs) -> DatasetType:
        return TorchTensorDatasetType(
            shape=(None,) + obj.shape[1:],
            dtype=str(obj.dtype)[len(obj.dtype.__module__) + 1 :],
        )


class TorchModelIO(ModelIO):
    """
    :class:`.ModelIO` implementation for PyTorch models
    """

    type: ClassVar[str] = "torch_io"
    is_jit: bool = False

    def dump(self, storage: Storage, path, model) -> Artifacts:
        self.is_jit = isinstance(model, torch.jit.ScriptModule)
        save = torch.jit.save if self.is_jit else torch.save
        with storage.open(path) as (f, art):
            save(model, f)
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

    @classmethod
    def process(
        cls, obj: Any, sample_data: Optional[Any] = None, **kwargs
    ) -> ModelType:
        model = TorchModel(model=obj, methods={})
        model.methods = {
            PREDICT_METHOD_NAME: Signature.from_method(
                model.predict,
                auto_infer=sample_data is not None,
                data=sample_data,
            ),
            "torch_predict": Signature.from_method(
                obj.__call__, auto_infer=sample_data is None, data=sample_data
            ),
        }
        return model

    def predict(self, data):
        if isinstance(data, torch.Tensor):
            return self.model(data)
        return self.model(*data)

    def get_requirements(self) -> Requirements:
        return super().get_requirements() + InstallableRequirement.from_module(
            mod=torch
        )


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
