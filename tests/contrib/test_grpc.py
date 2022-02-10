from typing import Any, ClassVar, Dict, List, Type

import pytest
from pydantic import BaseModel, conlist, parse_obj_as

from mlem.contrib.grpc import (
    GRPCField,
    GRPCMessage,
    create_message_from_type,
    create_messages,
)
from mlem.core.dataset_type import (
    DatasetSerializer,
    DatasetType,
    DatasetWriter,
)
from mlem.core.model import Signature
from mlem.core.requirements import Requirements


class SimpleModel(BaseModel):
    messages: ClassVar = [
        GRPCMessage(
            name="SimpleModel",
            fields=(GRPCField(rule="", type_="int", key="field", id_=1),),
        )
    ]
    field: int


class SimpleList(BaseModel):
    messages: ClassVar = [
        GRPCMessage(
            name="SimpleList",
            fields=(
                GRPCField(rule="repeated", type_="int", key="field", id_=1),
            ),
        )
    ]
    field: List[int]


class NestedModel(BaseModel):
    messages: ClassVar = [
        GRPCMessage(
            name="NestedModel",
            fields=(
                GRPCField(rule="", type_="SimpleModel", key="field", id_=1),
            ),
        )
    ] + SimpleModel.messages
    field: SimpleModel


class NestedList(BaseModel):
    messages: ClassVar = [
        GRPCMessage(
            name="NestedList",
            fields=(
                GRPCField(
                    rule="repeated", type_="SimpleModel", key="field", id_=1
                ),
            ),
        )
    ] + SimpleModel.messages
    field: List[SimpleModel]


class DoubleList(BaseModel):
    messages: ClassVar = [
        GRPCMessage(
            name="DoubleList",
            fields=(
                GRPCField(
                    rule="repeated",
                    type_="DoubleList_field",
                    key="field",
                    id_=1,
                ),
            ),
        ),
        GRPCMessage(
            name="DoubleList_field",
            fields=(
                GRPCField(rule="repeated", type_="int", key="__root__", id_=1),
            ),
        ),
    ]
    field: List[List[int]]


class ConSimpleList(BaseModel):
    messages: ClassVar = [
        GRPCMessage(
            name="ConSimpleList",
            fields=(
                GRPCField(rule="repeated", type_="int", key="field", id_=1),
            ),
        )
    ]
    field: conlist(int)  # type: ignore


class ConDoubleList(BaseModel):
    messages: ClassVar = [
        GRPCMessage(
            name="ConDoubleList",
            fields=(
                GRPCField(
                    rule="repeated",
                    type_="ConDoubleList_field",
                    key="field",
                    id_=1,
                ),
            ),
        ),
        GRPCMessage(
            name="ConDoubleList_field",
            fields=(
                GRPCField(rule="repeated", type_="int", key="__root__", id_=1),
            ),
        ),
    ]
    field: conlist(conlist(int))  # type: ignore


class SimpleDict(BaseModel):
    messages: ClassVar = ...
    field: Dict[str, int]


@pytest.mark.parametrize(
    "case",
    [
        SimpleModel,
        SimpleList,
        NestedModel,
        NestedList,
        DoubleList,
        ConSimpleList,
        ConDoubleList,
        SimpleDict,
    ],
)
def test_cases(case):
    messages = {}
    create_message_from_type(case, messages)
    assert set(messages.values()) == set(case.messages)


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
    messages = create_messages(Signature(name="aaa", args=[], returns=dt), {})
    repr = "\n".join(m.to_proto() for m in messages.values())
    print(repr)
    assert set(messages.values()) == set(expected)


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
                        type_="NumpyNdarray___root__",
                        key="__root__",
                        id_=1,
                    ),
                ),
            ),
            GRPCMessage(
                name="NumpyNdarray___root__",
                fields=(
                    GRPCField(
                        rule="repeated", type_="float", key="__root__", id_=1
                    ),
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
                        type_="NumpyNdarray___root__",
                        key="__root__",
                        id_=1,
                    ),
                ),
            ),
            GRPCMessage(
                name="NumpyNdarray___root__",
                fields=(
                    GRPCField(
                        rule="repeated",
                        type_="NumpyNdarray___root_____root__",
                        key="__root__",
                        id_=1,
                    ),
                ),
            ),
            GRPCMessage(
                name="NumpyNdarray___root_____root__",
                fields=(
                    GRPCField(
                        rule="repeated", type_="float", key="__root__", id_=1
                    ),
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
                        rule="",
                        type_="float",
                        key="sepal length (cm)",
                        id_=1,
                    ),
                    GRPCField(
                        rule="", type_="float", key="sepal width (cm)", id_=2
                    ),
                    GRPCField(
                        rule="",
                        type_="float",
                        key="petal length (cm)",
                        id_=3,
                    ),
                    GRPCField(
                        rule="", type_="float", key="petal width (cm)", id_=4
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
    _check_dt_messages(
        dt,
        [
            GRPCMessage(
                name="ContainerModel",
                fields=(
                    GRPCField(
                        rule="repeated",
                        type_="ContainerModel_field",
                        key="field",
                        id_=1,
                    ),
                ),
            ),
            GRPCMessage(
                name="ContainerModel_field",
                fields=(
                    GRPCField(
                        rule="repeated",
                        type_="ContainerModel_field___root__",
                        key="__root__",
                        id_=1,
                    ),
                ),
            ),
            GRPCMessage(
                name="ContainerModel_field___root__",
                fields=(
                    GRPCField(
                        rule="repeated", type_="int", key="__root__", id_=1
                    ),
                ),
            ),
        ],
    )
