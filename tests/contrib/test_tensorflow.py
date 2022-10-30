import os

import numpy as np
import pytest
import tensorflow as tf
from pydantic import parse_obj_as

from mlem.api import save
from mlem.constants import PREDICT_METHOD_NAME
from mlem.contrib.tensorflow import TFTensorDataType, TFTensorReader
from mlem.core.artifacts import LOCAL_STORAGE
from mlem.core.data_type import DataAnalyzer, DataType
from mlem.core.errors import DeserializationError, SerializationError
from mlem.core.model import ModelAnalyzer
from tests.conftest import data_write_read_check


@pytest.fixture
def np_data():
    return np.random.random((100, 32))


@pytest.fixture
def tensor_data(np_data):
    return tf.convert_to_tensor(np_data, dtype=tf.float32)


@pytest.fixture
def tftt(tensor_data):
    return DataAnalyzer.analyze(tensor_data)


@pytest.fixture
def tftt_3d(tensor_data):
    return DataAnalyzer.analyze(
        tf.tile(tf.expand_dims(tensor_data, -1), [1, 1, 20])
    )


def test_tensorflow_source(tensor_data):
    data = tensor_data
    data = DataType.create(data)

    def custom_assert(x, y):
        assert x.dtype == y.dtype
        assert isinstance(x, tf.Tensor)
        assert isinstance(y, tf.Tensor)

    data_write_read_check(
        data,
        custom_eq=tf.equal,
        reader_type=TFTensorReader,
        custom_assert=custom_assert,
    )


def test_feed_dict_type__self_serialization(tftt):
    assert isinstance(tftt, TFTensorDataType)
    assert tftt.get_requirements().modules == ["tensorflow"]
    payload = tftt.dict()
    tftt2 = parse_obj_as(TFTensorDataType, payload)
    assert tftt == tftt2


def test_feed_dict_type__serialization(tftt, tensor_data):
    payload = tftt.serialize(tensor_data)
    tensor_data2 = tftt.deserialize(payload)

    tf.assert_equal(tensor_data, tensor_data2)


@pytest.mark.parametrize(
    "obj",
    [
        1,  # wrong type
        tf.convert_to_tensor(
            np.random.random((100,)), dtype=tf.float32
        ),  # wrong rank
        tf.convert_to_tensor(
            np.random.random((100, 16)), dtype=tf.float32
        ),  # wrong shape
        tf.convert_to_tensor(
            np.random.random((100, 32)), dtype=tf.float64
        ),  # wrong value type
    ],
)
def test_feed_dict_serialize_failure(tftt, obj):
    with pytest.raises(SerializationError):
        tftt.serialize(obj)


@pytest.mark.parametrize(
    "obj",
    [
        1,  # wrong type
        [1] * 32,  # wrong rank
        [[1] * 16] * 100,  # wrong shape
        [["1"] * 32] * 100,  # wrong value type
    ],
)
def test_feed_dict_deserialize_failure(tftt, obj):
    with pytest.raises(DeserializationError):
        tftt.deserialize(obj)


def test_feed_dict_type__openapi_schema_3d(tftt_3d):
    assert tftt_3d.dict() == {
        "shape": (None, 32, 20),
        "dtype": "float32",
        "type": "tf_tensor",
    }
    assert tftt_3d.get_model().schema() == {
        "title": "TFTensor",
        "type": "array",
        "items": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 20,
                "maxItems": 20,
            },
            "minItems": 32,
            "maxItems": 32,
        },
    }


@pytest.fixture
def labels():
    return np.random.random((100, 10))


@pytest.fixture
def simple_net(np_data, labels):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(64, activation="relu"))
    model.add(tf.keras.layers.Dense(10))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.01),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    model.fit(np_data, labels, epochs=1, batch_size=50)

    return model


@pytest.fixture
def complex_net(np_data, labels):
    class Net(tf.keras.Model):
        def __init__(self):
            super().__init__(self)
            self.l1 = tf.keras.layers.Dense(50, activation="tanh")
            self.clf = tf.keras.layers.Dense(10, activation="relu")

        def call(self, inputs):
            return self.clf(self.l1(inputs))

    model = Net()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.01),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    model.fit(np_data, labels, epochs=1, batch_size=50)

    return model


@pytest.mark.parametrize(
    "net, input_data",
    [
        ("simple_net", "np_data"),
        ("simple_net", "tensor_data"),
        ("complex_net", "np_data"),
        ("complex_net", "tensor_data"),
    ],
)
def test_model_wrapper(net, input_data, tmpdir, request):
    net = request.getfixturevalue(net)
    input_data = request.getfixturevalue(input_data)

    orig_pred = net(input_data) if callable(net) else net.predict(input_data)

    tmw = ModelAnalyzer.analyze(net, sample_data=input_data)

    assert tmw.model is net

    expected_requirements = {"keras", "tensorflow", "numpy"}
    assert set(tmw.get_requirements().modules) == expected_requirements

    prediction = tmw.call_method("predict", input_data)

    np.testing.assert_allclose(orig_pred, prediction)

    model_name = str(tmpdir / "tensorflow-model")
    artifacts = tmw.dump(LOCAL_STORAGE, model_name)

    assert (
        os.path.isfile(model_name)
        if isinstance(net, tf.keras.Sequential)
        else os.path.isdir(model_name)
    )

    tmw.model = None
    with pytest.raises(ValueError):
        tmw.call_method(PREDICT_METHOD_NAME, input_data)

    tmw.load(artifacts)
    assert tmw.model is not net

    prediction2 = tmw.call_method("predict", input_data)

    np.testing.assert_allclose(prediction, prediction2)

    assert set(tmw.get_requirements().modules) == expected_requirements

    save(net, str(tmpdir / "tensorflow-net"), sample_data=input_data)


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
