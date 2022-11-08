import numpy as np
import pandas as pd
import pytest

from mlem.contrib.sklearn import SklearnModel
from mlem.core.objects import MlemModel
from mlem.core.requirements import Requirements
from mlem.runtime.interface import ModelInterface


class PandasModel:
    def __init__(self, prediction):
        self.prediction = prediction

    def predict(self, X: "pd.DataFrame"):
        assert isinstance(X, pd.DataFrame)
        return self.prediction


@pytest.fixture
def data():
    return pd.DataFrame([{"a": 1, "b": 1}])


@pytest.fixture
def prediction(data):
    return np.array([[0.5 for _ in range(data.size)]])


@pytest.fixture
def pd_model(data, prediction):
    return MlemModel(
        model_type=SklearnModel.process(
            PandasModel(prediction), sample_data=data
        ),
        requirements=Requirements.new(),
    )


def test_interface_types(pd_model: MlemModel, data, prediction):
    interface = ModelInterface.from_model(pd_model)
    # assert interface.exposed_method_docs('predict') == pd_model.description
    # TODO: https://github.com/iterative/mlem/issues/43
    pred = interface.execute("predict", {"X": data})
    assert (pred == prediction).all()


def test_with_serde(pd_model: MlemModel):
    interface = ModelInterface.from_model(pd_model)

    obj = {"values": [{"a": 1, "b": 1}]}

    data_type = pd_model.model_type.methods["predict"].args[0].type_
    data = data_type.get_serializer().deserialize(obj)

    interface.execute("predict", {"X": data})
