import pytest
from fastapi.testclient import TestClient

from mlem.contrib.fastapi import FastAPIServer
from mlem.core.objects import ModelMeta
from mlem.runtime.client.base import HTTPClient
from mlem.runtime.interface.base import ModelInterface


@pytest.fixture
def interface(model, train):
    model = ModelMeta.from_obj(model, sample_data=train)
    interface = ModelInterface.from_model(model)
    return interface


@pytest.fixture
def client(interface):
    app = FastAPIServer().app_init(interface)
    return TestClient(app)


@pytest.fixture
def request_mock(mocker, client):
    def patched_get(url, params=None, **kwargs):
        url = url[len("http://") :]
        return client.get(url, params=params, **kwargs)

    return mocker.patch(
        "mlem.runtime.client.base.requests.get",
        side_effect=patched_get,
    )


@pytest.fixture
def mlem_client(request_mock):
    return HTTPClient(host="", port=None)


def test_interface_endpoint(mlem_client):
    print(mlem_client.methods["predict"])
