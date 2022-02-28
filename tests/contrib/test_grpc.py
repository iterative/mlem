from typing import Any, ClassVar, Dict, List, Type

import pytest
from pydantic import BaseModel, conlist, parse_obj_as

from mlem.contrib.grpc import (
    GRPCField,
    GRPCList,
    GRPCMap,
    GRPCMessage,
    create_message_from_type,
    create_messages,
)
from mlem.core.dataset_type import (
    DatasetSerializer,
    DatasetType,
    DatasetWriter,
    PrimitiveType,
)
from mlem.core.model import Signature
from mlem.core.requirements import Requirements


class SimpleModel(BaseModel):
    messages: ClassVar = [
        GRPCMessage(
            name="SimpleModel",
            fields=(GRPCField(type_="int", field_name="field", id_=1),),
        )
    ]
    field: int


class SimpleList(BaseModel):
    messages: ClassVar = [
        GRPCMessage(
            name="SimpleList",
            fields=(GRPCList(type_="int", field_name="field", id_=1),),
        )
    ]
    field: List[int]


class NestedModel(BaseModel):
    messages: ClassVar = [
        GRPCMessage(
            name="NestedModel",
            fields=(
                GRPCField(type_="SimpleModel", field_name="field", id_=1),
            ),
        )
    ] + SimpleModel.messages
    field: SimpleModel


class NestedList(BaseModel):
    messages: ClassVar = [
        GRPCMessage(
            name="NestedList",
            fields=(GRPCList(type_="SimpleModel", field_name="field", id_=1),),
        )
    ] + SimpleModel.messages
    field: List[SimpleModel]


class DoubleList(BaseModel):
    messages: ClassVar = [
        GRPCMessage(
            name="DoubleList",
            fields=(
                GRPCList(
                    type_="DoubleList_field",
                    field_name="field",
                    id_=1,
                ),
            ),
        ),
        GRPCMessage(
            name="DoubleList_field",
            fields=(GRPCList(type_="int", field_name="__root__", id_=1),),
        ),
    ]
    field: List[List[int]]


class ConSimpleList(BaseModel):
    messages: ClassVar = [
        GRPCMessage(
            name="ConSimpleList",
            fields=(GRPCList(type_="int", field_name="field", id_=1),),
        )
    ]
    field: conlist(int)  # type: ignore


class ConDoubleList(BaseModel):
    messages: ClassVar = [
        GRPCMessage(
            name="ConDoubleList",
            fields=(
                GRPCList(
                    type_="ConDoubleList_field",
                    field_name="field",
                    id_=1,
                ),
            ),
        ),
        GRPCMessage(
            name="ConDoubleList_field",
            fields=(GRPCList(type_="int", field_name="__root__", id_=1),),
        ),
    ]
    field: conlist(conlist(int))  # type: ignore


class SimpleDict(BaseModel):
    messages: ClassVar = [
        GRPCMessage(
            name="SimpleDict",
            fields=(
                GRPCMap(
                    type_="str",
                    value_type="int",
                    field_name="field",
                    id_=1,
                ),
            ),
        ),
    ]
    field: Dict[str, int]


class ListOfDicts(BaseModel):
    messages: ClassVar = [
        GRPCMessage(
            name="ListOfDicts",
            fields=(
                GRPCList(
                    type_="ListOfDicts_field",
                    field_name="field",
                    id_=1,
                ),
            ),
        ),
        GRPCMessage(
            name="ListOfDicts_field",
            fields=(
                GRPCMap(
                    type_="str",
                    value_type="int",
                    field_name="__root__",
                    id_=1,
                ),
            ),
        ),
    ]
    field: List[Dict[str, int]]


class DictWithComplexValueType(BaseModel):
    class ComplexValue(BaseModel):
        x: int
        y: List[float]

    messages: ClassVar = [
        GRPCMessage(
            name="DictWithComplexValueType",
            fields=(
                GRPCMap(
                    type_="str",
                    value_type="ComplexValue",
                    field_name="field",
                    id_=1,
                ),
            ),
        ),
        GRPCMessage(
            name="ComplexValue",
            fields=(
                GRPCField(
                    type_="int",
                    field_name="x",
                    id_=1,
                ),
                GRPCList(
                    type_="float",
                    field_name="y",
                    id_=2,
                ),
            ),
        ),
    ]

    field: Dict[str, ComplexValue]


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
        ListOfDicts,
        DictWithComplexValueType,
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
                    GRPCList(
                        type_="NumpyNdarray___root__",
                        field_name="__root__",
                        id_=1,
                    ),
                ),
            ),
            GRPCMessage(
                name="NumpyNdarray___root__",
                fields=(
                    GRPCList(type_="float", field_name="__root__", id_=1),
                ),
            ),
        ],
    )


def test_lightgbm_numpy():
    from mlem.contrib.lightgbm import LightGBMDatasetType
    from mlem.contrib.numpy import NumpyNdarrayType

    dt = LightGBMDatasetType(
        inner=NumpyNdarrayType(shape=(None, 1), dtype="float64")
    )
    _check_dt_messages(
        dt,
        [
            GRPCMessage(
                name="NumpyNdarray",
                fields=(
                    GRPCList(
                        type_="NumpyNdarray___root__",
                        field_name="__root__",
                        id_=1,
                    ),
                ),
            ),
            GRPCMessage(
                name="NumpyNdarray___root__",
                fields=(
                    GRPCList(
                        type_="float",
                        field_name="__root__",
                        id_=1,
                    ),
                ),
            ),
        ],
    )


def test_lightgbm_pandas():
    from mlem.contrib.lightgbm import LightGBMDatasetType
    from mlem.contrib.pandas import DataFrameType

    dt = LightGBMDatasetType(
        inner=DataFrameType(columns=["a"], dtypes=["int64"], index_cols=[])
    )
    _check_dt_messages(
        dt,
        [
            GRPCMessage(
                name="DataFrame",
                fields=(
                    GRPCList(
                        type_="DataFrameRow",
                        field_name="values",
                        id_=1,
                    ),
                ),
            ),
            GRPCMessage(
                name="DataFrameRow",
                fields=(
                    GRPCField(
                        type_="int",
                        field_name="a",
                        id_=1,
                    ),
                ),
            ),
        ],
    )


def test_numpy_array():
    from mlem.contrib.numpy import NumpyNdarrayType

    dt = NumpyNdarrayType(shape=(3, 3, 3), dtype="float64")
    _check_dt_messages(
        dt,
        [
            GRPCMessage(
                name="NumpyNdarray",
                fields=(
                    GRPCList(
                        type_="NumpyNdarray___root__",
                        field_name="__root__",
                        id_=1,
                    ),
                ),
            ),
            GRPCMessage(
                name="NumpyNdarray___root__",
                fields=(
                    GRPCList(
                        type_="NumpyNdarray___root_____root__",
                        field_name="__root__",
                        id_=1,
                    ),
                ),
            ),
            GRPCMessage(
                name="NumpyNdarray___root_____root__",
                fields=(
                    GRPCList(type_="float", field_name="__root__", id_=1),
                ),
            ),
        ],
    )


def test_xgboost_dmatrix():
    from mlem.contrib.xgboost import DMatrixDatasetType

    dt = DMatrixDatasetType(
        is_from_list=False, feature_type_names=["int"], feature_names=["a"]
    )
    _check_dt_messages(
        dt,
        [
            GRPCMessage(
                name="DMatrixDataset",
                fields=(
                    GRPCField(type_="bool", field_name="is_from_list", id_=1),
                    GRPCList(
                        type_="str",
                        field_name="feature_type_names",
                        id_=2,
                    ),
                    GRPCList(
                        type_="str",
                        field_name="feature_names",
                        id_=3,
                    ),
                ),
            ),
        ],
    )


@pytest.mark.parametrize(
    "ptype", PrimitiveType.PRIMITIVES - {type(None), complex}
)
def test_primitive(ptype):
    dt = PrimitiveType(ptype=ptype.__name__)
    _check_dt_messages(
        dt,
        [
            GRPCMessage(
                name="Primitive",
                fields=(
                    GRPCField(
                        type_=ptype.__name__,
                        field_name="__root__",
                        id_=1,
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
                    GRPCList(
                        type_="DataFrameRow",
                        field_name="values",
                        id_=1,
                    ),
                ),
            ),
            GRPCMessage(
                name="DataFrameRow",
                fields=(
                    GRPCField(
                        type_="float",
                        field_name="sepal length (cm)",
                        id_=1,
                    ),
                    GRPCField(
                        type_="float", field_name="sepal width (cm)", id_=2
                    ),
                    GRPCField(
                        type_="float",
                        field_name="petal length (cm)",
                        id_=3,
                    ),
                    GRPCField(
                        type_="float", field_name="petal width (cm)", id_=4
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
                    GRPCList(
                        type_="ContainerModel_field",
                        field_name="field",
                        id_=1,
                    ),
                ),
            ),
            GRPCMessage(
                name="ContainerModel_field",
                fields=(
                    GRPCList(
                        type_="ContainerModel_field___root__",
                        field_name="__root__",
                        id_=1,
                    ),
                ),
            ),
            GRPCMessage(
                name="ContainerModel_field___root__",
                fields=(GRPCList(type_="int", field_name="__root__", id_=1),),
            ),
        ],
    )
