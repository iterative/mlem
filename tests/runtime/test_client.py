import platform

import numpy as np
import pytest
from fastapi.testclient import TestClient

from mlem.constants import PREDICT_ARG_NAME, PREDICT_METHOD_NAME
from mlem.contrib.fastapi import FastAPIServer
from mlem.contrib.numpy import NumpyNdarrayType
from mlem.core.dataset_type import DatasetAnalyzer
from mlem.core.errors import WrongMethodError
from mlem.core.model import Argument, Signature
from mlem.core.objects import ModelMeta
from mlem.runtime.client.base import HTTPClient
from mlem.runtime.interface.base import ModelInterface


@pytest.fixture
def signature(train):
    data_type = DatasetAnalyzer.analyze(train)
    returns_type = NumpyNdarrayType(
        shape=(None,),
        dtype="int32" if platform.system() == "Windows" else "int64",
    )
    kwargs = {"varkw": None}
    return Signature(
        name=PREDICT_METHOD_NAME,
        args=[Argument(name=PREDICT_ARG_NAME, type_=data_type)],
        returns=returns_type,
        **kwargs,
    )


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
def request_get_mock(mocker, client):
    def patched_get(url, params=None, **kwargs):
        url = url[len("http://") :]
        return client.get(url, params=params, **kwargs)

    return mocker.patch(
        "mlem.runtime.client.base.requests.get",
        side_effect=patched_get,
    )


@pytest.fixture
def request_post_mock(mocker, client):
    def patched_post(url, data=None, json=None, **kwargs):
        url = url[len("http://") :]
        return client.post(url, data=data, json=json, **kwargs)

    return mocker.patch(
        "mlem.runtime.client.base.requests.post",
        side_effect=patched_post,
    )


@pytest.fixture
def mlem_client(request_get_mock, request_post_mock):
    client = HTTPClient(host="", port=None)
    return client


@pytest.mark.parametrize("port", [None, 80])
def test_mlem_client_base_url(port):
    client = HTTPClient(host="", port=port)
    assert client.base_url == f"http://:{port}" if port else "http://"


@pytest.mark.parametrize("use_keyword", [False, True])
def test_interface_endpoint(mlem_client, train, signature, use_keyword):
    assert PREDICT_METHOD_NAME in mlem_client.methods
    assert mlem_client.methods[PREDICT_METHOD_NAME] == signature
    if use_keyword:
        assert np.array_equal(
            getattr(mlem_client, PREDICT_METHOD_NAME)(data=train),
            np.array([0] * 50 + [1] * 50 + [2] * 50),
        )
    else:
        assert np.array_equal(
            getattr(mlem_client, PREDICT_METHOD_NAME)(train),
            np.array([0] * 50 + [1] * 50 + [2] * 50),
        )


def test_wrong_endpoint(mlem_client):
    with pytest.raises(WrongMethodError):
        mlem_client.dummy_method()


def test_data_validation_more_params_than_expected(mlem_client, train):
    with pytest.raises(ValueError) as e:
        getattr(mlem_client, PREDICT_METHOD_NAME)(train, 2)
    assert str(e.value) == "Too much parameters given, expected: 1"


def test_data_validation_params_in_positional_and_keyword(mlem_client, train):
    with pytest.raises(ValueError) as e:
        getattr(mlem_client, f"sklearn_{PREDICT_METHOD_NAME}")(
            train, check_input=False
        )
    assert (
        str(e.value)
        == "Parameters should be passed either in positional or in keyword fashion, not both"
    )


def test_data_validation_params_with_wrong_name(mlem_client, train):
    with pytest.raises(ValueError) as e:
        getattr(mlem_client, PREDICT_METHOD_NAME)(X=train)
    assert (
        str(e.value)
        == 'Parameter with name "data" (position 0) should be passed'
    )
