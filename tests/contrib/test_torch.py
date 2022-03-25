import os

import pytest
import torch

from mlem.constants import PREDICT_METHOD_NAME
from mlem.contrib.torch import TorchModelIO
from mlem.core.artifacts import LOCAL_STORAGE
from mlem.core.dataset_type import DatasetAnalyzer
from mlem.core.errors import DeserializationError, SerializationError
from mlem.core.model import ModelAnalyzer


@pytest.fixture
def first_tensor():
    return torch.ones(5, 5, dtype=torch.int32)


@pytest.fixture
def second_tensor():
    return torch.rand(5, 10, dtype=torch.float32)


@pytest.fixture
def tdt_list(first_tensor, second_tensor):
    tensor_list = [first_tensor, second_tensor]
    return DatasetAnalyzer.analyze(tensor_list)


def test_torch_single_tensor(first_tensor):
    tdt = DatasetAnalyzer.analyze(first_tensor)

    assert tdt.get_requirements().modules == ["torch"]
    assert tdt.shape == (None, 5)
    assert tdt.dtype == "int32"

    tensor_deser = tdt.deserialize(tdt.serialize(first_tensor))
    assert torch.equal(first_tensor, tensor_deser)
    assert first_tensor.dtype == tensor_deser.dtype


def test_torch_tensors_list(tdt_list, first_tensor, second_tensor):
    assert tdt_list.get_requirements().modules == ["torch"]
    assert len(tdt_list.items) == 2
    assert tdt_list.items[0].shape == (None, 5)
    assert tdt_list.items[0].dtype == "int32"
    assert tdt_list.items[1].shape == (None, 10)
    assert tdt_list.items[1].dtype == "float32"

    tensor_list = [first_tensor, second_tensor]
    tensor_list_deser = tdt_list.deserialize(tdt_list.serialize(tensor_list))
    assert len(tensor_list) == len(tensor_list_deser)
    assert all(
        torch.equal(tensor, tensor_deser)
        and tensor.dtype == tensor_deser.dtype
        for tensor, tensor_deser in zip(tensor_list, tensor_list_deser)
    )


def test_torch_serialize_failure(tdt_list, first_tensor, second_tensor):
    objs = [
        first_tensor,  # not a list
        [first_tensor, second_tensor] * 2,  # not a list of 2
        [first_tensor] * 2,  # wrong dtype for second
        [first_tensor, first_tensor.float()],  # wrong shape for second
    ]

    for obj in objs:
        with pytest.raises(SerializationError):
            tdt_list.serialize(obj)


@pytest.mark.parametrize(
    "obj",
    [
        [[[1, 2], [3]], [[1], [2]]],  # illegal tensor for first
        [[[1, 2]], []],  # wrong shapes for both
    ],
)
def test_torch__deserialize_failure(tdt_list, obj):
    with pytest.raises(DeserializationError):
        tdt_list.deserialize(obj)


class MyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [torch.nn.Linear(5, 1), torch.nn.Linear(10, 1)]

    def forward(self, *inputs):
        results = torch.cat(
            [layer(input) for layer, input in zip(self.layers, inputs)], dim=1
        )
        return results.sum(dim=1)


def test_torch_empty_artifact_load_should_fail():
    with pytest.raises(ValueError):
        tmio = TorchModelIO()
        tmio.load([])


@pytest.mark.parametrize(
    "net", [torch.nn.Linear(5, 1), torch.jit.script(torch.nn.Linear(5, 1))]
)
def test_torch_builtin_net(net, first_tensor, tmpdir):
    check_model(net, first_tensor.float(), tmpdir)


def test_torch_custom_net(first_tensor, second_tensor, tmpdir):
    check_model(MyNet(), [first_tensor.float(), second_tensor], tmpdir)


def check_model(net, input_data, tmpdir):
    tmw = ModelAnalyzer.analyze(net, sample_data=input_data)
    assert tmw.model is net
    assert set(tmw.get_requirements().modules) == {"torch"}

    prediction = tmw.call_method("predict", input_data)

    model_name = str(tmpdir / "torch-model")
    artifacts = tmw.dump(LOCAL_STORAGE, model_name)
    assert os.path.isfile(model_name)

    tmw.model = None
    with pytest.raises(ValueError):
        tmw.call_method(PREDICT_METHOD_NAME, input_data)

    tmw.load(artifacts)
    assert tmw.model is not net

    prediction2 = tmw.call_method("predict", input_data)
    assert torch.equal(prediction, prediction2)
    assert set(tmw.get_requirements().modules) == {"torch"}


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
