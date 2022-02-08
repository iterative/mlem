from typing import Any, ClassVar, List, Type

import pytest
from pydantic import BaseModel, parse_obj_as

from mlem.contrib.grpc import GRPCField, GRPCMessage, create_messages
from mlem.core.dataset_type import (
    DatasetSerializer,
    DatasetType,
    DatasetWriter,
)
from mlem.core.model import Signature
from mlem.core.requirements import Requirements


@pytest.fixture
def interface():
    import pandas as pd
    from sklearn.datasets import load_iris
    from sklearn.tree import DecisionTreeClassifier

    from mlem.core.objects import ModelMeta
    from mlem.runtime.interface.base import ModelInterface

    train, target = load_iris(return_X_y=True, as_frame=True)
    train = pd.DataFrame(train)
    model = DecisionTreeClassifier().fit(train, target)
    return ModelInterface.from_model(
        ModelMeta.from_obj(model, sample_data=train)
    )


def _check_dt_messages(dt, expected):
    messages = create_messages(Signature(name="aaa", args=[], returns=dt))
    repr = "\n".join(m.to_proto() for m in messages)
    print(repr)
    assert messages == expected


def test_predict_proba(interface):
    dt = interface.get_method_returns("predict_proba")

    _check_dt_messages(
        dt,
        [
            GRPCMessage(
                name="NumpyNdarray",
                fields=(
                    GRPCField(
                        rule="repeated",
                        type_="ConstrainedListValue",
                        key="__root__",
                        id_=1,
                    ),
                ),
            ),
            GRPCMessage(
                name="ConstrainedListValue",
                fields=(
                    GRPCField(rule="repeated", type_="float", key="_", id_=1),
                ),
            ),
        ],
    )


def test_numpy():
    from mlem.contrib.numpy import NumpyNdarrayType

    dt = NumpyNdarrayType(shape=(3, 3, 3), dtype="float64")
    _check_dt_messages(
        dt,
        [
            GRPCMessage(
                name="NumpyNdarray",
                fields=(
                    GRPCField(
                        rule="repeated",
                        type_="ConstrainedListValue",
                        key="__root__",
                        id_=1,
                    ),
                ),
            ),
            GRPCMessage(
                name="ConstrainedListValue",
                fields=(
                    GRPCField(
                        rule="repeated",
                        type_="ConstrainedListValue_list",
                        key="_",
                        id_=1,
                    ),
                ),
            ),
            GRPCMessage(
                name="ConstrainedListValue_list",
                fields=(
                    GRPCField(rule="repeated", type_="float", key="_", id_=1),
                ),
            ),
        ],
    )


def test_predict(interface):
    dt = interface.get_method_args("predict")["data"]
    _check_dt_messages(
        dt,
        [
            GRPCMessage(
                name="DataFrame",
                fields=(
                    GRPCField(
                        rule="repeated",
                        type_="DataFrameRow",
                        key="values",
                        id_=1,
                    ),
                ),
            ),
            GRPCMessage(
                name="DataFrameRow",
                fields=(
                    GRPCField(
                        rule=None,
                        type_="float",
                        key="sepal length (cm)",
                        id_=1,
                    ),
                    GRPCField(
                        rule=None, type_="float", key="sepal width (cm)", id_=2
                    ),
                    GRPCField(
                        rule=None,
                        type_="float",
                        key="petal length (cm)",
                        id_=3,
                    ),
                    GRPCField(
                        rule=None, type_="float", key="petal width (cm)", id_=4
                    ),
                ),
            ),
        ],
    )


def test_container():
    class ContainerModel(BaseModel):
        field: List[List[List[int]]]

    class ContainerType(DatasetType, DatasetSerializer):
        type: ClassVar[str] = "test_container"

        def serialize(self, instance: Any) -> dict:
            return instance.dict()

        def deserialize(self, obj: dict) -> Any:
            return parse_obj_as(ContainerModel, obj)

        def get_requirements(self) -> Requirements:
            return Requirements.new()

        def get_writer(self, **kwargs) -> DatasetWriter:
            raise NotImplementedError

        def get_model(self) -> Type[BaseModel]:
            return ContainerModel

    dt = ContainerType()
    _check_dt_messages(dt, [])
