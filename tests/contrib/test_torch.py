import pytest
import torch

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


@pytest.mark.parametrize(
    "net", [torch.nn.Linear(5, 1), torch.jit.script(torch.nn.Linear(5, 1))]
)
def test_torch_builtin_net(net, first_tensor):
    check_model(net, first_tensor.float())


def test_torch_custom_net(first_tensor, second_tensor):
    check_model(MyNet(), [first_tensor.float(), second_tensor])


def check_model(net, input_data):
    tmw = ModelAnalyzer.analyze(net, sample_data=input_data)
    assert tmw.model is net
    assert set(tmw.get_requirements().modules) == {"torch"}
