from typing import Dict, List, Optional, Type
from pydantic import BaseModel, ConstrainedList
from mlem.core.model import Signature
from mlem.runtime import Interface
import typing_inspect
import inspect

class GRPCField(BaseModel):
    rule: Optional[str] = None
    type_: str
    key: str
    id_: int

    def to_expr(self):
        if self.rule:
            return f"{self.rule} {self.type_} {self.key} = {self.id_}"
        else:
            return f"{self.type_} {self.key} = {self.id_}"


class GRPCMessage(BaseModel):
    name: str
    fields: List[GRPCField]

    def to_proto(self):
        p_fields = "\n".join(each_field.to_expr() for each_field in self.fields)
        return f"""message {self.name} {{
            {p_fields}
        }}"""


def create_message_from_constraint_list(type_, list_models=None, suffix=None, message_name=None, type_name=None):
    if list_models is None:
        list_models = []
    if suffix is None:
        suffix = "_list"
    if message_name is None:
        message_name = type_.__name__
    if type_name is None:
        type_name = type_.__name__

    inner_type = type_.item_type
    if issubclass(inner_type, ConstrainedList):
        type_name+=suffix
        message_name+=suffix
        field = GRPCField(rule="repeated", type_=type_name, key="_", id_=1)
        list_models.extend(create_message_from_constraint_list(inner_type, list_models, suffix, message_name, type_name))
        message_name = message_name[:-len(suffix)]
    else:
        field = GRPCField(rule="repeated", type_=inner_type.__name__, key="_", id_=1)

    message = GRPCMessage(name=message_name, fields=[field])
    return [message] + list_models


def create_message_from_generic(type_, message_name, generic_models=None):
    if generic_models is None:
        generic_models = []
    generic_type = type_.__origin__
    inner_type = type_.__args__[0]

    if generic_type is list:
        field_rule = "repeated"
        suffix = "_list"
    
    if typing_inspect.is_generic_type(inner_type):
        message_name+=suffix
        field = GRPCField(rule=field_rule, type_=message_name, key="_", id_=1)
        generic_models.extend(create_message_from_generic(inner_type, message_name, generic_models))
        message_name = message_name[:-len(suffix)]
    else:
        field = GRPCField(rule=field_rule, type_=inner_type.__name__, key="_", id_=1)
    
    message = GRPCMessage(name=message_name, fields=[field])
    return [message] + generic_models


def get_models_recursively(model: BaseModel, field_models=None):
    if field_models is None:
        field_models = []
    message_name = model.__name__
    all_fields = []
    num_fields = 0
    for each_field_key, each_field_val in model.__fields__.items():
        field_rule = None
        outer_type_ = each_field_val.outer_type_
        type_ = each_field_val.type_
        # if typing_inspect.is_generic_type(outer_type_):
        #     generic_models = create_message_from_generic(outer_type_, message_name=each_field_val.name)
        #     field_models.extend(generic_models)
        if inspect.isclass(type_) and issubclass(type_, ConstrainedList):
            list_models = create_message_from_constraint_list(type_)
            field_models.extend(list_models)
        if inspect.isclass(type_) and issubclass(type_, BaseModel):
            field_models.extend(get_models_recursively(type_, field_models))
        field_rule = "repeated" if outer_type_ != type_ and outer_type_._name == "List" else None
        field_type = type_.__name__
        num_fields+=1
        field = GRPCField(rule=field_rule, type_=field_type, key=each_field_key, id_=num_fields)
        all_fields.append(field)
    if all_fields:
        message = GRPCMessage(name=message_name, fields=all_fields)
        return [message] + field_models
    else:
        return field_models


def create_messages(signature: Signature):
    models: Dict[BaseModel] = {}
    returns_model = signature.returns.get_serializer().get_model()
    returns_field_models = get_models_recursively(returns_model)
    args_field_models = []
    for arg in signature.args:
        args_field_models.extend(get_models_recursively(arg.type_.get_serializer().get_model()))
    models = returns_field_models + args_field_models
    return [each_model.to_proto() for each_model in models]


def create_method_definition(method_name, signature: Signature):
    args = [arg.type_.get_serializer().get_model().__name__ for arg in signature.args]
    returns = signature.returns.get_serializer().get_model().__name__
    return f"""rpc {method_name} ({", ".join(args)}) returns ({returns}) {{}}"""


def create_grpc_protobuf(interface: Interface):
    service_name = interface.model_type.model.__class__.__name__
    rpc_definitions = []
    rpc_messages = []
    for method_name, signature in interface.iter_methods():
        rpc_definitions.append(create_method_definition(method_name, signature))
        rpc_messages.extend(create_messages(signature))

    rpc = "\n".join(each_rpc for each_rpc in rpc_definitions)
    service_proto = f"""service {service_name} {{
        {rpc}
    }}"""
    return service_proto, rpc_messages


if __name__ == '__main__':
    from sklearn.datasets import load_iris
    from sklearn.tree import DecisionTreeClassifier
    import pandas as pd
    from mlem.core.objects import ModelMeta
    from mlem.runtime.interface.base import ModelInterface
    from mlem.core.dataset_type import DatasetType, DatasetWriter, DatasetSerializer
    from mlem.core.requirements import Requirements
    from typing import Any, ClassVar

    class Container(DatasetType, DatasetSerializer):
        __name__ = "container"
        type: ClassVar[str] = "test_container"
        field: List[List[List[int]]]

        def serialize(self, instance: Any) -> dict:
            return instance

        def deserialize(self, obj: dict) -> Any:
            return obj

        def get_requirements(self) -> Requirements:
            return Requirements.new([])

        def get_writer(self, **kwargs) -> DatasetWriter:
            raise NotImplementedError()
        
        def get_serializer(
            self, **kwargs  # pylint: disable=unused-argument
        ) -> "DatasetSerializer":
            if isinstance(self, DatasetSerializer):
                return self
            raise NotImplementedError

        def get_model(self) -> Type[BaseModel]:
            return self

    train, target = load_iris(return_X_y=True, as_frame=True)
    train = pd.DataFrame(train)
    model = DecisionTreeClassifier().fit(train, target)
    interface = ModelInterface.from_model(ModelMeta.from_obj(model, sample_data=train))

    dt = interface.get_method_returns("predict_proba")
    # dt = NumpyNdarrayType(shape=(3,3,3), dtype="float64")
    # dt  = interface.get_method_args("predict")['data']
    # dt = Container(field=[[[3,3,3]]])

    print("\n".join(list(set(create_messages(Signature(name="aaa", args=[], returns=dt))))))
    # print(dt.get_serializer().get_model().schema())
    # rpc_definitions, rpc_messages = create_grpc_protobuf(interface)
    # print(rpc_definitions)
    # print(list(set(rpc_messages)))
