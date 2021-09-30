import shlex
from functools import wraps
from typing import Any, Dict, List, Optional, Type, TypeVar

import click
from pydantic import parse_obj_as
from yaml import safe_load

from mlem.core.base import MlemObject
from mlem.core.meta_io import deserialize
from mlem.core.metadata import load_meta

OBJ_CTX_NAME = "model_meta"


def model_meta_from_ctx(*params, pass_context=False):
    def outer(f):
        @click.pass_context
        @wraps(f)
        def inner(ctx, *args, **kwargs):
            meta = ctx.obj[OBJ_CTX_NAME]
            if not pass_context:
                return f(meta, *args, **kwargs)
            return f(ctx, meta, *args, **kwargs)

        return inner

    if callable(params[0]):
        outer = outer(params[0])
    return outer


def model_meta_to_ctx(f):
    @click.argument("model")
    @click.pass_context
    @wraps(f)
    def inner(ctx, model, *args, **kwargs):
        ctx.ensure_object(dict)
        meta = load_meta(model)
        ctx.obj[OBJ_CTX_NAME] = meta
        return f(*args, **kwargs)

    return inner


def with_model_meta(f):
    @click.argument("model")
    @wraps(f)
    def inner(model, **kwargs):
        meta = load_meta(model)
        return f(model=meta, **kwargs)

    return inner


def _set_recursively(obj: dict, keys: List[str], value: Any):
    if len(keys) == 1:
        obj[keys[0]] = value
        return
    key, keys = keys[0], keys[1:]
    if key not in obj:
        obj[key] = {}
    _set_recursively(obj[key], keys, value)


def smart_split(string: str, char: str):
    SPECIAL = "\0"
    if char != " ":
        string = string.replace(" ", SPECIAL).replace(char, " ")
    return [
        s.replace(" ", char).replace(SPECIAL, " ") for s in shlex.split(string)
    ]


def build_model(
    model: Type[MlemObject],
    subtype: str,
    conf: List[str],
    file_conf: List[str],
):
    model_dict = {model.__type_field__: subtype}

    for file in file_conf:
        keys, path = smart_split(file, "=")
        with open(path, "r") as f:
            value = safe_load(f)
        _set_recursively(model_dict, smart_split(keys, "."), value)

    for c in conf:
        keys, value = smart_split(c, "=")
        _set_recursively(model_dict, smart_split(keys, "."), value)
    return parse_obj_as(model, model_dict)


def config_arg(name: str, model: Type[MlemObject], **kwargs):
    """add argument + multi option -c and -f to configure and deserialize to model"""

    def decorator(f):
        @click.option("-l", "--load", default=None)
        @click.argument("subtype", default="", **kwargs)
        @click.option("-c", "--conf", multiple=True)
        @click.option("-f", "--file_conf", multiple=True)
        @wraps(f)
        def inner(
            load: Optional[str],
            subtype: str,
            conf: List[str],
            file_conf: List[str],
            **inner_kwargs,
        ):
            if load is not None:
                with open(load, "r") as of:
                    obj = deserialize(safe_load(of), model)
            else:
                obj = build_model(model, subtype, conf, file_conf)
            inner_kwargs[name] = obj
            return f(**inner_kwargs)

        return inner

    return decorator


T = TypeVar("T", bound=MlemObject)


def create_configurable(
    cls: Type[T], kind: str, class_mapping: Dict[str, str] = None
) -> T:
    if class_mapping is not None:
        kind = class_mapping.get(kind, kind)
    args = {"type": kind}
    clazz = cls.resolve_subtype(kind)
    for _name, field in clazz.__fields__.items():
        try:
            cast = field.type_
            default = field.default
            args[field.name] = cast(
                click.prompt(f"{field.name} value?", default=default)
            )
        except ValueError:
            raise NotImplementedError(
                f"Not yet implemented for type {field.type_}"
            )
    return deserialize(args, cls)
