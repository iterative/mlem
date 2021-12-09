"""
Here we test that functions local imports are also collected with another level of indirection
"""
import six  # pylint: disable=unused-import # noqa
from sklearn.linear_model import LinearRegression

LR = LinearRegression()


def model(data):
    from proxy_pkg_import import (  # pylint: disable=unused-import # noqa
        pkg_func,
    )

    pkg_func()
    assert hasattr(LR, "predict")
    return data
