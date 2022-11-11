import platform

import numpy as np
import pytest

from mlem.constants import PREDICT_ARG_NAME, PREDICT_METHOD_NAME
from mlem.contrib.numpy import NumpyNdarrayType
from mlem.core.data_type import DataAnalyzer
from mlem.core.errors import WrongMethodError
from mlem.core.model import Argument, Signature
from mlem.runtime.client import HTTPClient


@pytest.fixture
def signature(train):
    data_type = DataAnalyzer.analyze(train)
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
    with pytest.raises(
        ValueError,
        match="Parameters should be passed either in positional or in keyword fashion, not both",
    ):
        getattr(mlem_client, PREDICT_METHOD_NAME)(train, check_input=False)


def test_data_validation_params_with_wrong_name(mlem_client, train):
    with pytest.raises(ValueError) as e:
        getattr(mlem_client, PREDICT_METHOD_NAME)(X=train)
    assert (
        str(e.value)
        == 'Parameter with name "data" (position 0) should be passed'
    )
