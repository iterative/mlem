import inspect
from typing import Dict, List, Optional
from pydantic import BaseModel
from mlem.core.model import Signature
from mlem.runtime import Interface


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


def get_models_recursively(model: BaseModel, field_models=[]):
    message_name = model.__name__
    all_fields = []
    num_fields = 0
    for each_field_key in model.__fields__:
        field_rule = None
        each_field_val = model.__fields__[each_field_key]
        outer_type_ = each_field_val.outer_type_
        type_ = each_field_val.type_
        if BaseModel in inspect.getmro(type_):
            field_models.extend(get_models_recursively(type_, field_models))
        if outer_type_ != type_:
            collection_type = outer_type_._name
            if collection_type == "List":
                field_rule = "repeated"
        field_type = type_.__name__
        field_key = each_field_key
        num_fields+=1
        field = GRPCField(rule=field_rule, type_=field_type, key=field_key, id_=num_fields)
        all_fields.append(field)
    message = GRPCMessage(name=message_name, fields=all_fields)
    return [message] + field_models


def create_message(signature: Signature):
    models: Dict[BaseModel] = {}
    returns_model = signature.returns.get_serializer().get_model()
    returns_field_models = get_models_recursively(returns_model, [])
    args_field_models = []
    for each_arg in signature.args:
        args_field_models.extend(get_models_recursively(each_arg.type_.get_serializer().get_model(), []))
    models = returns_field_models + args_field_models
    return [each_model.to_proto() for each_model in models]


def create_method_definition(method_name, signature: Signature):
    args = [each_arg.type_.get_serializer().get_model().__name__ for each_arg in signature.args]
    returns = signature.returns.get_serializer().get_model().__name__
    return f"""rpc {method_name} ({", ".join(args)}) returns ({returns}) {{}}"""


def create_grpc_protobuf(interface: Interface):
    service_name = interface.model_type.model.__class__.__name__
    rpc_definitions = []
    rpc_messages = []
    for method_name, signature in interface.iter_methods():
        rpc_definitions.append(create_method_definition(method_name, signature))
        rpc_messages.extend(create_message(signature))

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

    train, target = load_iris(return_X_y=True)
    train = pd.DataFrame(train)
    model = DecisionTreeClassifier().fit(train, target)
    interface = ModelInterface.from_model(ModelMeta.from_obj(model, sample_data=train))
    rpc_definitions, rpc_messages = create_grpc_protobuf(interface)
    print(rpc_definitions)
    print(list(set(rpc_messages)))
