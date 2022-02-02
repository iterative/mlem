# pylint: disable=W0212

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel, create_model
from pydantic.main import ModelMetaclass

from mlem.constants import PREDICT_ARG_NAME, PREDICT_METHOD_NAME
from mlem.contrib.fastapi import FastAPIServer, rename_recursively
from mlem.contrib.numpy import NumpyNdarrayType
from mlem.core.dataset_type import DatasetAnalyzer
from mlem.core.model import Argument, Signature
from mlem.core.objects import ModelMeta
from mlem.runtime.interface.base import ModelInterface


@pytest.fixture
def signature(train):
    data_type = DatasetAnalyzer.analyze(train)
    returns_type = NumpyNdarrayType(shape=(None,), dtype="float64")
    kwargs = {"varkw": "kwargs"}
    return Signature(
        name=PREDICT_METHOD_NAME,
        args=[Argument(name=PREDICT_ARG_NAME, type_=data_type)],
        returns=returns_type,
        **kwargs,
    )


@pytest.fixture
def payload_model(signature):
    serializers = {
        arg.name: arg.type_.get_serializer() for arg in signature.args
    }
    kwargs = {
        key: (serializer.get_model(), ...)
        for key, serializer in serializers.items()
    }
    return create_model("Model", **kwargs)


@pytest.fixture
def interface(model, train):
    model = ModelMeta.from_obj(model, sample_data=train)
    interface = ModelInterface.from_model(model)
    return interface


@pytest.fixture
def executor(interface):
    return interface.get_method_executor(PREDICT_METHOD_NAME)


@pytest.fixture
def client(interface):
    app = FastAPIServer().app_init(interface)
    return TestClient(app)


def test_rename_recursively(payload_model):
    rename_recursively(payload_model, "predict")

    def recursive_assert(model):
        assert model.__name__.startswith("predict")
        for field in model.__fields__.values():
            if issubclass(field.type_, BaseModel):
                recursive_assert(field.type_)

    recursive_assert(payload_model)


def test_create_handler(signature, executor):
    server = FastAPIServer()
    _, response_model = server._create_handler(
        PREDICT_METHOD_NAME, signature, executor
    )
    assert (
        response_model.__name__
        == f"{PREDICT_METHOD_NAME}{signature.returns.get_serializer().get_model().__name__}"
    )
    assert isinstance(response_model, ModelMetaclass)


def test_endpoint(client, interface, train):
    payload = (
        interface.model_type.methods[PREDICT_METHOD_NAME]
        .args[0]
        .type_.get_serializer()
        .serialize(train)
    )
    response = client.post(
        f"/{PREDICT_METHOD_NAME}",
        json={"data": payload},
    )
    assert response.status_code == 200
    assert response.json() == [0] * 50 + [1] * 50 + [2] * 50
