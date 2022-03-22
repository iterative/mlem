import os
import posixpath
import tempfile
from typing import Any, ClassVar, Optional, Tuple

import torch

from mlem.constants import PREDICT_METHOD_NAME
from mlem.core.artifacts import Artifacts, Storage
from mlem.core.dataset_type import DatasetHook, DatasetType, DatasetWriter
from mlem.core.errors import DeserializationError, SerializationError
from mlem.core.hooks import TOP_PRIORITY_VALUE, IsInstanceHookMixin
from mlem.core.model import ModelHook, ModelIO, ModelType, Signature
from mlem.core.requirements import InstallableRequirement, Requirements


class TorchTensorDatasetType(DatasetType, DatasetHook, IsInstanceHookMixin):
    """
    :class:`.DatasetType` implementation for `torch.Tensor` objects
    which converts them to built-in Python lists and vice versa.

    :param shape: shape of `torch.Tensor` objects in dataset
    :param dtype: data type of `torch.Tensor` objects in dataset
    """

    type: ClassVar[str] = "torch"
    valid_types: ClassVar = (torch.Tensor,)
    shape: Tuple
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
            raise DeserializationError(
                f"given object: {obj} could not be converted to tensor "
                f"of type: {getattr(torch, self.dtype)}"
            )
        self._check_shape(ret, DeserializationError)
        return ret

    def get_requirements(self) -> Requirements:
        return Requirements.new([InstallableRequirement.from_module(torch)])

    def get_writer(self, **kwargs) -> DatasetWriter:
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
    model_file_name = "model.pth"
    model_jit_file_name = "model.jit.pth"

    def dump(self, storage: Storage, path, model) -> Artifacts:
        is_jit = isinstance(model, torch.jit.ScriptModule)
        save = torch.jit.save if is_jit else torch.save
        model_name = (
            self.model_jit_file_name if is_jit else self.model_file_name
        )
        with tempfile.TemporaryDirectory(prefix="mlem_torch_dump") as f:
            model_path = os.path.join(f, model_name)
            save(model, model_path)
            fs_path = posixpath.join(path, model_name)
            return [storage.upload(model_path, fs_path)]

    def load(self, artifacts: Artifacts):
        if len(artifacts) != 1:
            raise ValueError(
                f"Invalid artifacts: should be one of {self.model_file_name} OR {self.model_jit_file_name} file"
            )

        with tempfile.TemporaryDirectory(prefix="mlem_torch_load") as tmpdir:
            local_path = os.path.join(tmpdir, self.model_jit_file_name)
            load = torch.jit.load
            if not os.path.exists(local_path):
                local_path = os.path.join(tmpdir, self.model_file_name)
                load = torch.load
            artifacts[0].materialize(
                local_path,
            )
            return load(model_file=local_path)


class TorchModel(ModelType, ModelHook, IsInstanceHookMixin):
    """
    :class:`.ModelType` implementation for PyTorch models
    """

    type: ClassVar[str] = "torch"
    valid_types: ClassVar = (torch.nn.Module,)
    io: ModelIO = TorchModelIO()
    priority: ClassVar = TOP_PRIORITY_VALUE

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
