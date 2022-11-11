"""Tensorflow models support
Extension type: model

ModelType and ModelIO implementations for `tf.keras.Model`
DataType, Reader and Writer implementations for `tf.Tensor`
"""
import posixpath
import tempfile
from typing import Any, ClassVar, Iterator, List, Optional, Tuple

import h5py
import numpy as np
import tensorflow as tf
from pydantic import conlist, create_model
from tensorflow.python.keras.saving.saved_model_experimental import sequential

from mlem.contrib.numpy import python_type_from_np_string_repr
from mlem.core.artifacts import Artifacts, Storage
from mlem.core.data_type import (
    DataHook,
    DataReader,
    DataSerializer,
    DataType,
    DataWriter,
)
from mlem.core.errors import DeserializationError, SerializationError
from mlem.core.hooks import IsInstanceHookMixin
from mlem.core.model import (
    BufferModelIO,
    ModelHook,
    ModelIO,
    ModelType,
    Signature,
)
from mlem.core.requirements import InstallableRequirement, Requirements


def python_type_from_tf_string_repr(dtype: str):
    #  not sure this will work all the time
    return python_type_from_np_string_repr(dtype)


class TFTensorDataType(
    DataType, DataSerializer, DataHook, IsInstanceHookMixin
):
    """
    DataType implementation for `tensorflow.Tensor`
    """

    type: ClassVar[str] = "tf_tensor"
    valid_types: ClassVar = (tf.Tensor,)
    shape: Tuple[Optional[int], ...]
    """Shape of `tensorflow.Tensor` objects in data"""
    dtype: str
    """Data type of `tensorflow.Tensor` objects in data"""

    @property
    def tf_type(self):
        return getattr(tf, self.dtype)

    def _check_shape(self, tensor, exc_type):
        if tuple(tensor.shape)[1:] != self.shape[1:]:
            raise exc_type(
                f"given tensor is of shape: {(None,) + tuple(tensor.shape)[1:]}, expected: {self.shape}"
            )

    def serialize(self, instance: tf.Tensor):
        self.check_type(instance, tf.Tensor, SerializationError)
        if instance.dtype is not self.tf_type:
            raise SerializationError(
                f"given tensor is of dtype: {instance.dtype}, "
                f"expected: {self.tf_type}"
            )
        self._check_shape(instance, SerializationError)
        return instance.numpy().tolist()

    def deserialize(self, obj):
        try:
            ret = tf.convert_to_tensor(obj, dtype=self.tf_type)
        except (ValueError, TypeError):
            raise DeserializationError(  # pylint: disable=raise-missing-from
                f"given object: {obj} could not be converted to tensor "
                f"of type: {self.tf_type}"
            )
        self._check_shape(ret, DeserializationError)
        return ret

    def get_requirements(self) -> Requirements:
        return Requirements.new([InstallableRequirement.from_module(tf)])

    def get_writer(
        self, project: str = None, filename: str = None, **kwargs
    ) -> DataWriter:
        return TFTensorWriter(**kwargs)

    def _subtype(self, subshape: Tuple[Optional[int], ...]):
        if len(subshape) == 0:
            return python_type_from_tf_string_repr(self.dtype)
        return conlist(
            self._subtype(subshape[1:]),
            min_items=subshape[0],
            max_items=subshape[0],
        )

    def get_model(self, prefix: str = ""):
        return create_model(
            prefix + "TFTensor",
            __root__=(List[self._subtype(self.shape[1:])], ...),  # type: ignore
        )

    @classmethod
    def process(cls, obj: tf.Tensor, **kwargs) -> DataType:
        return TFTensorDataType(
            shape=(None,) + tuple(obj.shape)[1:],
            dtype=obj.dtype.name,
        )


DATA_KEY = "data"


class TFTensorWriter(DataWriter):
    """Write tensorflow tensors to np format"""

    type: ClassVar[str] = "tf_tensor"

    def write(
        self, data: DataType, storage: Storage, path: str
    ) -> Tuple[DataReader, Artifacts]:
        with storage.open(path) as (f, art):
            np.savez_compressed(f, **{DATA_KEY: data.data.numpy()})
        return TFTensorReader(data_type=data), {self.art_name: art}


class TFTensorReader(DataReader):
    """Read tensorflow tensors from np format"""

    type: ClassVar[str] = "tf_tensor"

    def read(self, artifacts: Artifacts) -> DataType:
        if DataWriter.art_name not in artifacts:
            raise ValueError(
                f"Wrong artifacts {artifacts}: should be one {DataWriter.art_name} file"
            )
        with artifacts[DataWriter.art_name].open() as f:
            np_data = np.load(f)[DATA_KEY]
            data = tf.convert_to_tensor(
                np_data, dtype=getattr(tf, np_data.dtype.name)
            )
            return self.data_type.copy().bind(data)

    def read_batch(
        self, artifacts: Artifacts, batch_size: int
    ) -> Iterator[DataType]:
        raise NotImplementedError


def is_custom_net(model):
    return (
        not model._is_graph_network  # pylint:disable=protected-access
        and not isinstance(model, sequential.Sequential)
    )


class TFKerasModelIO(BufferModelIO):
    """
    IO for Tensorflow Keras models (:class:`tensorflow.keras.Model` objects)
    """

    type: ClassVar[str] = "tf_keras"
    save_format: Optional[str] = None
    """`tf` for custom net classes and `h5` otherwise"""

    def save_model(self, model: tf.keras.Model, path: str):
        if self.save_format is None:
            self.save_format = "tf" if is_custom_net(model) else "h5"
        model.save(path, save_format=self.save_format)

    def load(  # pylint:disable=inconsistent-return-statements
        self, artifacts: Artifacts
    ):
        if self.save_format == "h5":
            if self.art_name not in artifacts:
                raise ValueError(
                    "Invalid artifacts: should have only one file"
                )
            with artifacts[self.art_name].open() as f:
                return tf.keras.models.load_model(h5py.File(f))

        if self.save_format == "tf":
            with tempfile.TemporaryDirectory() as tmpdir:
                for k, a in artifacts.items():
                    a.materialize(posixpath.join(tmpdir, k))
                return tf.keras.models.load_model(tmpdir)
        else:
            raise ValueError(
                f"Unknown save format {self.save_format} for tensorflow models, expected one of [tf, h5]"
            )


class TFKerasModel(ModelType, ModelHook, IsInstanceHookMixin):
    """
    :class:`.ModelType` implementation for Tensorflow Keras models
    """

    type: ClassVar[str] = "tf_keras"
    valid_types: ClassVar = (tf.keras.Model,)
    io: ModelIO = TFKerasModelIO()
    """IO"""

    @classmethod
    def process(
        cls, obj: Any, sample_data: Optional[Any] = None, **kwargs
    ) -> ModelType:
        model = TFKerasModel(model=obj, methods={})
        model.methods = {
            "predict": Signature.from_method(
                obj.__call__,
                sample_data,
                auto_infer=sample_data is not None,
            ),
        }
        return model

    def predict(self, data):
        return self.model(data)

    def get_requirements(self) -> Requirements:
        return super().get_requirements() + InstallableRequirement.from_module(
            mod=tf
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
