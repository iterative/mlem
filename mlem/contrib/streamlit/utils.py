from typing import Callable, Optional, Tuple, Type

import streamlit
import streamlit_pydantic
from pydantic import BaseModel

from mlem.cli.utils import LIST_LIKE_SHAPES
from mlem.runtime import InterfaceMethod
from mlem.runtime.client import HTTPClient
from mlem.runtime.interface import ExecutionError, InterfaceArgument


def augment_model(
    model: Type,
) -> Tuple[Callable, Optional[Type]]:
    if not issubclass(model, BaseModel):
        return lambda x: x, model
    list_model = [
        (name, f)
        for name, f in model.__fields__.items()
        if f.shape in LIST_LIKE_SHAPES
    ]

    if len(list_model) == 0:
        return lambda x: x, model

    if len(list_model) > 1:
        return lambda x: x, None
    name, field = list_model[0]

    return lambda x: model(**{name: [x]}), field.type_


def method_form(method_name: str, method: InterfaceMethod, client: HTTPClient):
    with streamlit.form(key=method_name):
        arg_values = method_args(method_name, method)
        submit_button = streamlit.form_submit_button(label="Submit")

    if submit_button:
        with streamlit.tabs(["Response:"])[0]:
            with streamlit.spinner("Processing..."):
                try:
                    response = call_method(client, method_name, arg_values)
                except ExecutionError as e:
                    streamlit.error(e)
                    return
            if method.returns.get_serializer().serializer.is_binary:
                streamlit.download_button(label="Download", data=response)
            else:
                streamlit.write(response)


def call_method(client: HTTPClient, method_name: str, arg_values: dict):
    return getattr(client, method_name)(
        **{
            k: v.dict() if isinstance(v, BaseModel) else v
            for k, v in arg_values.items()
        }
    )


def method_args(method_name: str, method: InterfaceMethod):
    arg_values = {}
    for arg in method.args:
        serializer = arg.get_serializer()
        if serializer.serializer.is_binary:
            arg_values[arg.name] = streamlit.file_uploader(arg.name)
        else:
            arg_model = serializer.get_model()
            augment, arg_model_aug = augment_model(arg_model)
            if arg_model_aug is None:
                streamlit.error("Model with lists/arrays are not supported")
                return None

            arg_values[arg.name] = augment(
                method_arg(
                    arg, arg_model_aug, f"{method_name}_{arg.name}_form"
                )
            )
    return arg_values


def method_arg(arg: InterfaceArgument, arg_model: Type, key: str):
    if issubclass(arg_model, BaseModel):
        return streamlit_pydantic.pydantic_input(key=key, model=arg_model)
    if arg_model in (int, float):
        arg_input = streamlit.number_input(
            key=key, label=arg.name, value=arg.default or 0
        )
    elif arg_model in (str, complex):
        arg_input = streamlit.text_input(
            key=key, label=arg.name, value=arg.default or ""
        )
    elif arg_model is bool:
        arg_input = streamlit.checkbox(
            key=key, label=arg.name, value=arg.default
        )
    else:
        arg_input = None
    return arg_input


def model_methods_tabs(client: HTTPClient):
    methods = list(client.methods.keys())
    tabs = streamlit.tabs(methods)

    for method_name, tab in zip(methods, tabs):
        with tab:
            tab.header(method_name)
            method: InterfaceMethod = client.methods[method_name]
            method_form(method_name, method, client)


def model_form(client: HTTPClient):
    methods = list(client.methods.keys())
    if len(methods) == 1:
        method_name = methods[0]
        method: InterfaceMethod = client.methods[method_name]
        method_form(method_name, method, client)
        return
    model_methods_tabs(client)
