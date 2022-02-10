import inspect
from typing import Dict, List, Optional, Tuple, Type

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

    rule: Optional[str] = None
    type_: str
    key: str
    id_: int

    def to_expr(self):
        if self.rule:
            return f"{self.rule} {self.type_} {self.key} = {self.id_}"
        return f"{self.type_} {self.key} = {self.id_}"


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
    field = GRPCField(
        rule="repeated",
        type_=create_message_from_type(
            inner_type, existing_messages, prefix + "___root__"
        ),
        key="__root__",
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
        field_rule = "repeated"
    else:
        raise NotImplementedError

    msg = GRPCMessage(
        name=prefix,
        fields=(
            (
                GRPCField(
                    rule=field_rule,
                    type_=create_message_from_type(
                        inner_type, existing_messages, prefix + "___root__"
                    ),
                    key="__root__",
                    id_=1,
                )
            ),
        ),
    )
    existing_messages[type_] = msg
    return msg.name


def _get_rule_from_outer_type(outer_type: Type) -> str:
    if getattr(outer_type, "__origin__", None) is list:
        return "repeated"
    if not typing_inspect.is_generic_type(outer_type):
        return ""
    raise NotImplementedError


def create_message_from_base_model(
    model: Type[BaseModel], existing_messages: MessageMapping, prefix: str = ""
) -> str:
    name = model.__name__
    fields = []
    for id_, (field_name, field_info) in enumerate(
        model.__fields__.items(), start=1
    ):
        outer_type_ = field_info.outer_type_
        rule = _get_rule_from_outer_type(outer_type_)
        type_ = field_info.type_

        fields.append(
            GRPCField(
                rule=rule,
                type_=create_message_from_type(
                    type_,
                    existing_messages,
                    prefix=f"{prefix}{name}_{field_name}",
                ),
                key=field_name,
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
