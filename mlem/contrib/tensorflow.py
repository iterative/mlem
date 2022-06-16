import os.path
import tempfile
from typing import Any, ClassVar, Iterator, Optional, Tuple

import h5py
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.saving.saved_model_experimental import sequential

from mlem.constants import PREDICT_METHOD_NAME
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


class TFTensorDataType(
    DataType, DataSerializer, DataHook, IsInstanceHookMixin
):
    """
    :class:`.DataType` implementation for `tensorflow.Tensor` objects
    which converts them to built-in Python lists and vice versa.

    :param shape: shape of `tensorflow.Tensor` objects in data
    :param dtype: data type of `tensorflow.Tensor` objects in data
    """

    type: ClassVar[str] = "tensorflow"
    valid_types: ClassVar = (tf.Tensor,)
    shape: Tuple[Optional[int], ...]
    dtype: str

    def _check_shape(self, tensor, exc_type):
        if tuple(tensor.shape)[1:] != self.shape[1:]:
            raise exc_type(
                f"given tensor is of shape: {(None,) + tuple(tensor.shape)[1:]}, expected: {self.shape}"
            )

    def serialize(self, instance: tf.Tensor):
        self.check_type(instance, tf.Tensor, SerializationError)
        if instance.dtype is not getattr(tf, self.dtype):
            raise SerializationError(
                f"given tensor is of dtype: {instance.dtype}, "
                f"expected: {getattr(tf, self.dtype)}"
            )
        self._check_shape(instance, SerializationError)
        return instance.numpy().tolist()

    def deserialize(self, obj):
        try:
            ret = tf.convert_to_tensor(obj, dtype=getattr(tf, self.dtype))
        except (ValueError, TypeError):
            raise DeserializationError(  # pylint: disable=raise-missing-from
                f"given object: {obj} could not be converted to tensor "
                f"of type: {getattr(tf, self.dtype)}"
            )
        self._check_shape(ret, DeserializationError)
        return ret

    def get_requirements(self) -> Requirements:
        return Requirements.new([InstallableRequirement.from_module(tf)])

    def get_writer(
        self, project: str = None, filename: str = None, **kwargs
    ) -> DataWriter:
        return TFTensorWriter(**kwargs)

    def get_model(self, prefix: str = ""):
        raise NotImplementedError

    @classmethod
    def process(cls, obj: tf.Tensor, **kwargs) -> DataType:
        return TFTensorDataType(
            shape=tuple(obj.shape),
            dtype=obj.dtype.name,
        )


DATA_KEY = "data"


class TFTensorWriter(DataWriter):
    type: ClassVar[str] = "tensorflow"

    def write(
        self, data: DataType, storage: Storage, path: str
    ) -> Tuple[DataReader, Artifacts]:
        with storage.open(path) as (f, art):
            np.savez_compressed(f, **{DATA_KEY: data.data.numpy()})
        return TFTensorReader(data_type=data), {self.art_name: art}


class TFTensorReader(DataReader):
    type: ClassVar[str] = "tensorflow"

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
    :class:`.ModelIO` implementation for Tensorflow Keras models (:class:`tensorflow.keras.Model` objects)
    """

    type: ClassVar[str] = "tensorflow_io"
    save_format: Optional[str] = None

    def save_model(self, model: tf.keras.Model, path: str):
        if self.save_format is None:
            self.save_format = "tf" if is_custom_net(model) else "h5"
        model.save(path, save_format=self.save_format)

    def load(  # pylint:disable=inconsistent-return-statements
        self, artifacts: Artifacts
    ):
        if self.save_format == "h5":
            if len(artifacts) != 1:
                raise ValueError(
                    "Invalid artifacts: should have only one file"
                )
            with artifacts[self.art_name].open() as f:
                return tf.keras.models.load_model(h5py.File(f))

        if self.save_format == "tf":
            with tempfile.TemporaryDirectory() as tmpdir:
                for k, a in artifacts.items():
                    a.materialize(os.path.join(tmpdir, k))
                return tf.keras.models.load_model(tmpdir)


class TFKerasModel(ModelType, ModelHook, IsInstanceHookMixin):
    """
    :class:`.ModelType` implementation for Tensorflow Keras models
    """

    type: ClassVar[str] = "tensorflow"
    valid_types: ClassVar = (tf.keras.Model,)
    io: ModelIO = TFKerasModelIO()

    @classmethod
    def process(
        cls, obj: Any, sample_data: Optional[Any] = None, **kwargs
    ) -> ModelType:
        model = TFKerasModel(model=obj, methods={})
        model.methods = {
            PREDICT_METHOD_NAME: Signature.from_method(
                model.predict,
                auto_infer=sample_data is not None,
                data=sample_data,
            ),
            "tensorflow_predict": Signature.from_method(
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
