import pytest
from fastapi.testclient import TestClient
from mlem.core.objects import ModelMeta
from mlem.runtime.interface.base import ModelInterface
from mlem.runtime.client.base import HTTPClient
from mlem.contrib.fastapi import FastAPIServer

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
def request_mock(mocker):
    def patched_get(url, params=None, **kwargs):
        url = url[len("http://"):]
        return client.get(url, params=params, **kwargs)

    return mocker.patch(
        f"mlem.runtime.client.base.requests.get",
        side_effect=patched_get,
    )

@pytest.fixture
def mlem_client(client, request_mock):
    return HTTPClient(host=client.base_url[len("http://"):], port=None)

def test_interface_endpoint(mlem_client):
    print(mlem_client.methods['predict'])
