# pylint: disable=W0212
from typing import Callable

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import create_model
from pydantic.main import BaseModel, ModelMetaclass

from mlem.constants import PREDICT_ARG_NAME, PREDICT_METHOD_NAME
from mlem.contrib.fastapi import (
    FastAPIMiddleware,
    FastAPIServer,
    rename_recursively,
)
from mlem.contrib.numpy import NumpyNdarrayType
from mlem.core.data_type import DataAnalyzer, FileSerializer
from mlem.core.model import Argument, Signature
from mlem.core.objects import MlemModel
from mlem.runtime.client import Client
from mlem.runtime.interface import Interface, InterfaceMethod, ModelInterface
from mlem.runtime.middleware import Middlewares
from mlem.runtime.server import ServerInterface
from tests.contrib.test_pandas import pandas_assert


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
    interface = ServerInterface.create(
        FastAPIServer(standardize=True), ModelInterface.from_model(model)
    )
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
    _, response_model, _ = server._create_handler(
        PREDICT_METHOD_NAME,
        InterfaceMethod.from_signature(f_signature),
        executor,
        server.middlewares,
    )
    assert (
        response_model.__name__
        == f"{PREDICT_METHOD_NAME}_response_{f_signature.returns.get_model().__name__}"
    )
    assert isinstance(response_model, ModelMetaclass)


def test_create_handler_primitive():
    def f(data):
        return data

    signature: InterfaceMethod = InterfaceMethod.from_signature(
        Signature.from_method(f, auto_infer=True, data="value")
    )

    server = FastAPIServer()
    handler, response_model, _ = server._create_handler(
        PREDICT_METHOD_NAME, signature, f, server.middlewares
    )
    request_model = handler.__annotations__["model"]
    assert request_model.__fields__["data"].type_ is str
    assert response_model is str
    assert handler(request_model(data="value")) == "value"


def test_endpoint(f_client, f_interface: Interface, create_mlem_client, train):
    docs = f_client.get("/openapi.json")
    assert docs.status_code == 200, docs.json()

    mlem_client: Client = create_mlem_client(f_client)
    remote_interface = mlem_client.interface
    payload_1 = (
        remote_interface.__root__[PREDICT_METHOD_NAME]
        .args[0]
        .get_serializer()
        .serialize(train)
    )

    payload = (
        f_interface.get_method_signature(PREDICT_METHOD_NAME)
        .args[0]
        .get_serializer()
        .serialize(train)
    )
    assert payload == payload_1

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
        interface.model.model_type.methods["__call__"]
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


@pytest.mark.parametrize(
    "data,eq_assert",
    [
        [np.array([1, 2]), np.testing.assert_allclose],
        [pd.DataFrame([{"col": 1}]), pandas_assert],
    ],
)
def test_file_endpoint(
    create_mlem_client, create_client, data, eq_assert: Callable, tmp_path
):
    model_interface = ModelInterface.from_model(
        MlemModel.from_obj(lambda x: x, sample_data=data)
    )

    server = FastAPIServer(
        standardize=False,
        request_serializer=FileSerializer(),
        response_serializer=FileSerializer(),
    )
    interface = ServerInterface.create(server, model_interface)
    client = create_client(server, interface)

    docs = client.get("/openapi.json")
    assert docs.status_code == 200, docs.json()

    mlem_client: Client = create_mlem_client(client)
    remote_interface = mlem_client.interface
    dt = remote_interface.__root__["__call__"].args[0].data_type
    ser = FileSerializer()
    with ser.dump(dt, data) as f:
        response = client.post("/__call__", files={"file": f})
    assert response.status_code == 200
    resp_array = ser.deserialize(dt, response.content)
    eq_assert(resp_array, data)

    eq_assert(mlem_client(data), data)

    path = tmp_path / "data"
    with open(path, "wb") as fout, ser.dump(dt, data) as fin:
        fout.write(fin.read())

    eq_assert(mlem_client(str(path)), data)
    eq_assert(mlem_client(path), data)


def test_serve_processors_model(
    processors_model, create_mlem_client, create_client
):
    model_interface = ModelInterface.from_model(processors_model)

    server = FastAPIServer(
        standardize=True,
    )
    interface = ServerInterface.create(server, model_interface)
    client = create_client(server, interface)

    docs = client.get("/openapi.json")
    assert docs.status_code == 200, docs.json()

    mlem_client: Client = create_mlem_client(client)
    remote_interface = mlem_client.interface
    dt = remote_interface.__root__["predict"].args[0].data_type
    response = client.post(
        "/predict", json={"data": dt.serialize(["1", "2", "3"])}
    )
    assert response.status_code == 200
    resp = remote_interface.__root__["predict"].returns.data_type.deserialize(
        response.json()
    )
    assert resp == 4


class TestFastAPIMiddleware(FastAPIMiddleware):
    add: int
    mult: int

    def on_app_init(self, app: FastAPI):
        app.add_api_route("/kek", lambda: "ok")

    def on_init(self):
        pass

    def on_request(self, request):
        request["data"] += self.add
        return request

    def on_response(self, request, response):
        return response * self.mult


def test_fastapi_middleware(create_mlem_client, create_client):
    model = MlemModel.from_obj(lambda x: x, sample_data=10)
    model_interface = ModelInterface.from_model(model)

    server = FastAPIServer(
        standardize=True,
        middlewares=Middlewares(
            __root__=[TestFastAPIMiddleware(add=1, mult=10)]
        ),
    )
    interface = ServerInterface.create(server, model_interface)
    client = create_client(server, interface)

    docs = client.get("/openapi.json")
    assert docs.status_code == 200, docs.json()

    kek = client.get("/kek")
    assert kek.status_code == 200
    assert kek.text == '"ok"'

    mlem_client: Client = create_mlem_client(client)
    remote_interface = mlem_client.interface
    dt = remote_interface.__root__["predict"].args[0].data_type
    response = client.post("/predict", json={"data": dt.serialize(1)})
    assert response.status_code == 200
    resp = remote_interface.__root__["predict"].returns.data_type.deserialize(
        response.json()
    )
    assert resp == 20
