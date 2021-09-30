import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB

from mlem.contrib.sklearn import SklearnModel
from mlem.core.model import ModelAnalyzer


@pytest.mark.parametrize(
    "mtype",
    [
        (GaussianNB, ["predict", "predict_proba"]),
        (LinearRegression, ["predict"]),
    ],
)
def test_sklearn_model(mtype):
    cls, methods = mtype
    data = np.array([[1], [2]])
    res = data[:, 0]
    print(data, res)
    model = cls().fit(data, res)
    assert SklearnModel.is_object_valid(model)
    mt = ModelAnalyzer.analyze(model, test_data=data)
    assert isinstance(mt, SklearnModel)
    assert set(mt.methods.keys()) == set(methods)
