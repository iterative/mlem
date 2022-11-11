# pylint: disable=W0212

import pytest
from fastapi.testclient import TestClient
from pydantic import create_model
from pydantic.main import BaseModel, ModelMetaclass

from mlem.constants import PREDICT_ARG_NAME, PREDICT_METHOD_NAME
from mlem.contrib.fastapi import FastAPIServer, rename_recursively
from mlem.contrib.numpy import NumpyNdarrayType
from mlem.core.data_type import DataAnalyzer
from mlem.core.model import Argument, Signature
from mlem.core.objects import MlemModel
from mlem.runtime.interface import (
    Interface,
    ModelInterface,
    prepare_model_interface,
)


@pytest.fixture
def f_signature(train):
    data_type = DataAnalyzer.analyze(train)
    returns_type = NumpyNdarrayType(shape=(None,), dtype="float64")
    kwargs = {"varkw": "kwargs"}
    return Signature(
        name=PREDICT_METHOD_NAME,
        args=[Argument(name=PREDICT_ARG_NAME, type_=data_type)],
        returns=returns_type,
        **kwargs,
    )


@pytest.fixture
def payload_model(f_signature):
    serializers = {
        arg.name: arg.type_.get_serializer() for arg in f_signature.args
    }
    kwargs = {
        key: (serializer.get_model(), ...)
        for key, serializer in serializers.items()
    }
    return create_model("Model", **kwargs)


@pytest.fixture
def f_interface(model, train):
    model = MlemModel.from_obj(model, sample_data=train)
    interface = prepare_model_interface(model, FastAPIServer(standardize=True))
    return interface


@pytest.fixture
def executor(f_interface):
    return f_interface.get_method_executor(PREDICT_METHOD_NAME)


@pytest.fixture
def f_client(f_interface):
    app = FastAPIServer(standardize=True).app_init(f_interface)
    return TestClient(app)


def test_rename_recursively(payload_model):
    rename_recursively(payload_model, "predict")

    def recursive_assert(model):
        assert model.__name__.startswith("predict")
        for field in model.__fields__.values():
            if issubclass(field.type_, BaseModel):
                recursive_assert(field.type_)

    recursive_assert(payload_model)


def test_create_handler(f_signature, executor):
    server = FastAPIServer()
    _, response_model = server._create_handler(
        PREDICT_METHOD_NAME, f_signature, executor
    )
    assert (
        response_model.__name__
        == f"{PREDICT_METHOD_NAME}_response_{f_signature.returns.get_serializer().get_model().__name__}"
    )
    assert isinstance(response_model, ModelMetaclass)


def test_create_handler_primitive():
    def f(data):
        return data

    signature = Signature.from_method(f, auto_infer=True, data="value")

    server = FastAPIServer()
    handler, response_model = server._create_handler(
        PREDICT_METHOD_NAME, signature, f
    )
    request_model = handler.__annotations__["model"]
    assert request_model.__fields__["data"].type_ is str
    assert response_model is str
    assert handler(request_model(data="value")) == "value"


def test_endpoint(f_client, f_interface: Interface, train):
    docs = f_client.get("/openapi.json")
    assert docs.status_code == 200, docs.json()
    payload = (
        f_interface.get_method_signature(PREDICT_METHOD_NAME)
        .args[0]
        .type_.get_serializer()
        .serialize(train)
    )
    response = f_client.post(
        f"/{PREDICT_METHOD_NAME}",
        json={"data": payload},
    )
    assert response.status_code == 200, response.text
    assert response.json() == [0] * 50 + [1] * 50 + [2] * 50


@pytest.mark.parametrize(
    "data",
    [
        [-1, 0, 1],
        [{"key": "value", "key2": 1}, {"key": "value", "key2": 1}],
        [{"key": "value", "key2": 1}],
        [{"key": "value"}],
        {"key": [1]},
        {"key": [1], "key2": [2]},
        {"key": [1, 2]},
        {"key": [1, 2], "key2": [3, 4]},
        [[1]],
        [[1, 1]],
        [[1], [2]],
        [[1, 1], [2, 2]],
        {"key": {"key": 1}},
        {"key": {"key": 1}, "key2": {"key": 1}},
        {"key": {"key": 1, "key2": 1}},
        {"key": {"key": 1, "key2": 1}, "key2": {"key": 1, "key2": 1}},
    ],
)
def test_nested_objects_in_schema(data):
    model = MlemModel.from_obj(lambda x: x, sample_data=data)
    interface = ModelInterface.from_model(model)

    app = FastAPIServer().app_init(interface)
    client = TestClient(app)

    docs = client.get("/openapi.json")
    assert docs.status_code == 200, docs.json()
    payload = (
        interface.model_type.methods["__call__"]
        .args[0]
        .type_.get_serializer()
        .serialize(data)
    )

    response = client.post(
        "/__call__",
        json={"x": payload},
    )
    assert response.status_code == 200, response.json()
    assert response.json() == data
