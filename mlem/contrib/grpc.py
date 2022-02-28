import inspect
from typing import Dict, List, Tuple, Type

import typing_inspect
from pydantic import BaseModel, ConstrainedList

from mlem.core.model import Signature
from mlem.runtime.interface.base import ModelInterface


class GRPCMethod(BaseModel):
    class Config:
        frozen = True

    name: str
    args: List[str]
    returns: str

    def to_proto(self):
        return f"""rpc {self.name} ({", ".join(self.args)}) returns ({self.returns}) {{}}"""


class GRPCService(BaseModel):
    name: str
    methods: List[GRPCMethod]

    def to_proto(self):
        rpc = "\n".join(each_rpc.to_proto() for each_rpc in self.methods)
        return f"""service {self.name} {{
            {rpc}
        }}"""


class GRPCField(BaseModel):
    class Config:
        frozen = True

    type_: str
    field_name: str
    id_: int

    def to_expr(self):
        return f"{self.type_} {self.field_name} = {self.id_}"


class GRPCMap(GRPCField):
    class Config:
        frozen = True

    value_type: str

    def to_expr(self):
        return f"map<{self.type_}, {self.value_type}> {self.field_name} = {self.id_}"


class GRPCList(GRPCField):
    class Config:
        frozen = True

    rule: str = "repeated"

    def to_expr(self):
        return f"{self.rule} {self.type_} {self.field_name} = {self.id_}"


class GRPCMessage(BaseModel):
    class Config:
        frozen = True

    name: str
    fields: Tuple[GRPCField, ...]

    def to_proto(self) -> str:
        p_fields = "\n".join(
            each_field.to_expr() for each_field in self.fields
        )
        return f"""message {self.name} {{
            {p_fields}
        }}"""


MessageMapping = Dict[Type, GRPCMessage]


def create_message_from_constrained_list(
    type_: Type[ConstrainedList],
    existing_messages: MessageMapping,
    prefix: str = "",
) -> str:
    inner_type = type_.item_type
    field = GRPCList(
        type_=create_message_from_type(
            inner_type, existing_messages, prefix + "___root__"
        ),
        field_name="__root__",
        id_=1,
    )
    message = GRPCMessage(name=prefix, fields=(field,))
    existing_messages[type_] = message
    return message.name


def create_message_from_generic(
    type_, existing_messages: MessageMapping, prefix: str = ""
) -> str:
    generic_type = type_.__origin__
    inner_type = type_.__args__[0]

    if generic_type is list:
        fields = (
            (
                GRPCList(
                    type_=create_message_from_type(
                        inner_type, existing_messages, prefix + "___root__"
                    ),
                    field_name="__root__",
                    id_=1,
                )
            ),
        )
    elif generic_type is dict:
        key_type, _ = get_keytype_and_valuetype_from_dict(type_)
        value_type = type_.__args__[1]
        fields = (
            (
                GRPCMap(
                    type_=key_type,
                    value_type=create_message_from_type(
                        value_type, existing_messages, prefix + "___root__"
                    ),
                    field_name="__root__",
                    id_=1,
                )
            ),
        )
    else:
        raise NotImplementedError

    msg = GRPCMessage(
        name=prefix,
        fields=fields,
    )
    existing_messages[type_] = msg
    return msg.name


def _get_rule_from_outer_type(outer_type: Type) -> str:
    if getattr(outer_type, "__origin__", None) is list:
        return "repeated"
    elif getattr(outer_type, "__origin__", None) is dict:
        return "map"
    if not typing_inspect.is_generic_type(outer_type):
        return ""
    raise NotImplementedError


def get_keytype_and_valuetype_from_dict(d):
    left_brace_position = d.__str__().find("[")
    comma_position = d.__str__().find(",")
    right_brace_position = d.__str__().find("]")
    key_type = d.__str__()[left_brace_position + 1 : comma_position].rsplit(
        "."
    )[-1]
    value_type = d.__str__()[comma_position + 2 : right_brace_position].rsplit(
        "."
    )[-1]
    return key_type, value_type


def create_message_from_base_model(
    model: Type[BaseModel], existing_messages: MessageMapping, prefix: str = ""
) -> str:
    name = model.__name__
    fields = []
    for id_, (field_name, field_info) in enumerate(
        model.__fields__.items(), start=1
    ):
        outer_type_ = field_info.outer_type_
        type_ = field_info.type_
        rule = _get_rule_from_outer_type(outer_type_)
        msg_sub_type = create_message_from_type(
            type_,
            existing_messages,
            prefix=f"{prefix}{name}_{field_name}",
        )

        if rule == "map":
            key_type, _ = get_keytype_and_valuetype_from_dict(outer_type_)
            fields.append(
                GRPCMap(
                    type_=key_type,
                    value_type=msg_sub_type,
                    field_name=field_name,
                    id_=id_,
                ),
            )
        elif rule == "repeated":
            fields.append(
                GRPCList(
                    type_=msg_sub_type,
                    field_name=field_name,
                    id_=id_,
                )
            )
        else:
            fields.append(
                GRPCField(
                    type_=msg_sub_type,
                    field_name=field_name,
                    id_=id_,
                )
            )
    message = GRPCMessage(name=name, fields=tuple(fields))
    existing_messages[model] = message
    return message.name


def create_message_from_type(
    type_: Type, existing_messages: MessageMapping, prefix: str = ""
) -> str:
    if type_ in existing_messages:
        return existing_messages[type_].name

    if type_ in (str, int, float, bool):
        return type_.__name__  # TODO: mapping
    if typing_inspect.is_generic_type(type_):
        return create_message_from_generic(type_, existing_messages, prefix)
    if issubclass(type_, BaseModel):
        return create_message_from_base_model(type_, existing_messages, prefix)
    if inspect.isclass(type_) and issubclass(type_, ConstrainedList):
        return create_message_from_constrained_list(
            type_, existing_messages, prefix
        )
    raise NotImplementedError


def create_messages(
    signature: Signature, messages: MessageMapping
) -> MessageMapping:
    returns_model = signature.returns.get_serializer().get_model()
    create_message_from_type(returns_model, messages)
    for arg in signature.args:
        create_message_from_type(
            arg.type_.get_serializer().get_model(), messages
        )
    return messages


def create_method_definition(method_name, signature: Signature) -> GRPCMethod:
    args = [
        arg.type_.get_serializer().get_model().__name__
        for arg in signature.args
    ]
    returns = signature.returns.get_serializer().get_model().__name__
    return GRPCMethod(name=method_name, args=args, returns=returns)


def create_grpc_protobuf(
    interface: ModelInterface,
) -> Tuple[
    GRPCService, List[GRPCMessage]
]:  # TODO: change to regular Interface and get name some other way
    service_name = interface.model_type.model.__class__.__name__
    methods = []
    messages: MessageMapping = {}
    for method_name, signature in interface.iter_methods():
        methods.append(create_method_definition(method_name, signature))
        create_messages(signature, messages)

    return GRPCService(name=service_name, methods=methods), list(
        messages.values()
    )
