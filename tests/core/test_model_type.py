import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB

from mlem.contrib.numpy import NumpyNdarrayType
from mlem.contrib.pandas import DataFrameType
from mlem.contrib.sklearn import SklearnModel
from mlem.core.dataset_type import UnspecifiedDatasetType
from mlem.core.model import ModelAnalyzer, Signature


@pytest.mark.parametrize(
    "mtype",
    [
        (
            GaussianNB,
            [
                "predict",
                "predict_proba",
                "sklearn_predict",
                "sklearn_predict_proba",
            ],
        ),
        (LinearRegression, ["predict", "sklearn_predict"]),
    ],
)
def test_sklearn_model(mtype):
    cls, methods = mtype
    data = np.array([[1], [2]])
    res = data[:, 0]
    print(data, res)
    model = cls().fit(data, res)
    assert SklearnModel.is_object_valid(model)
    mt = ModelAnalyzer.analyze(model, sample_data=data)
    assert isinstance(mt, SklearnModel)
    assert set(mt.methods.keys()) == set(methods)


def test_infer_signatire_unspecified(model):
    signature = Signature.from_method(model.predict)
    assert signature.name == "predict"
    assert signature.returns == UnspecifiedDatasetType()
    assert len(signature.args) == 2
    arg = signature.args[0]
    assert arg.name == "X"
    assert arg.type_ == UnspecifiedDatasetType()


def test_infer_signatire(model, train):
    signature = Signature.from_method(model.predict, auto_infer=True, X=train)
    assert signature.name == "predict"
    assert signature.returns == NumpyNdarrayType(shape=(None,), dtype="int64")
    assert len(signature.args) == 2
    arg = signature.args[0]
    assert arg.name == "X"
    assert arg.type_ in (  # dont know how to do this better
        DataFrameType(
            columns=["0", "1", "2", "3"],
            dtypes=["float64", "float64", "float64", "float64"],
            index_cols=[],
        ),
        NumpyNdarrayType(shape=(None, 4), dtype="float64"),
    )
