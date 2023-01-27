import io
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, BinaryIO, Callable, ClassVar, Optional

import requests
from pydantic import BaseModel, parse_obj_as

from mlem.core.base import MlemABC
from mlem.core.errors import MlemError, WrongMethodError
from mlem.runtime.interface import (
    ExecutionError,
    InterfaceDescriptor,
    InterfaceMethod,
    VersionedInterfaceDescriptor,
)

logger = logging.getLogger(__name__)


class Client(MlemABC, ABC):
    """Client is a way to invoke methods on running `Server` instance.
    `Client`s dynamically define python methods based on interfaces
    exposed by `Server`"""

    class Config:
        type_root = True
        type_field = "type"
        exclude = {"interface_cache"}

    type: ClassVar[str]
    abs_name: ClassVar[str] = "client"
    interface_cache: Optional[InterfaceDescriptor] = None
    raw: bool = False
    """Pass values as-is without serializers"""

    @property
    def interface(self) -> InterfaceDescriptor:
        if self.interface_cache is None:
            self.interface_cache = self._interface_factory()
        return self.interface_cache

    @property
    def methods(self):
        return self.interface.__root__

    @abstractmethod
    def _interface_factory(self) -> InterfaceDescriptor:
        raise NotImplementedError

    @abstractmethod
    def _call_method(self, name: str, args: Any, return_raw: bool):
        raise NotImplementedError

    @abstractmethod
    def _call_method_binary(self, name: str, arg: BinaryIO, return_raw: bool):
        raise NotImplementedError

    def __getattr__(self, name):
        if name not in self.methods:
            raise WrongMethodError(f"{name} method is not exposed by server")
        return _MethodCall(
            method=self.methods[name],
            name=name,
            call_method=self._call_method,
            call_method_binary=self._call_method_binary,
            raw=self.raw,
        )

    def __call__(self, *args, **kwargs):
        if "__call__" not in self.methods:
            raise WrongMethodError("__call__ method is not exposed by server")
        return _MethodCall(
            method=self.methods["__call__"],
            name="__call__",
            call_method=self._call_method,
            call_method_binary=self._call_method_binary,
            raw=self.raw,
        )(*args, **kwargs)


class _MethodCall(BaseModel):
    method: InterfaceMethod
    name: str
    call_method: Callable
    call_method_binary: Callable
    raw: bool

    # pylint: disable=too-many-branches
    def __call__(self, *args, **kwargs):
        if args and kwargs:
            raise ValueError(
                "Parameters should be passed either in positional or in keyword fashion, not both"
            )
        if len(args) > len(self.method.args) or len(kwargs) > len(
            self.method.args
        ):
            raise ValueError(
                f"Too much parameters given, expected: {len(self.method.args)}"
            )

        data = {}
        return_raw = self.method.returns.get_serializer().serializer.is_binary

        for i, arg in enumerate(self.method.args):
            obj = None
            if len(args) > i:
                obj = args[i]
            if obj is None:
                obj = kwargs.get(arg.name, None)
            if obj is None:
                raise ValueError(
                    f'Parameter with name "{arg.name}" (position {i}) should be passed'
                )

            serializer = arg.get_serializer()
            if serializer.serializer.is_binary:
                if len(self.method.args) > 1:
                    raise NotImplementedError(
                        "Multiple file requests are not supported yet"
                    )
                if (
                    isinstance(obj, (str, os.PathLike))
                    and serializer.support_files
                ):
                    with open(obj, "rb") as f:
                        return (
                            self.method.returns.get_serializer().deserialize(
                                self.call_method_binary(
                                    self.name, f, return_raw
                                )
                            )
                        )
                elif isinstance(obj, io.BytesIO):
                    return self.method.returns.get_serializer().deserialize(
                        self.call_method_binary(self.name, obj, return_raw)
                    )
                else:
                    with serializer.dump(obj) as f:
                        return (
                            self.method.returns.get_serializer().deserialize(
                                self.call_method_binary(
                                    self.name, f, return_raw
                                )
                            )
                        )
            if self.raw:
                data[arg.name] = obj
            else:
                data[arg.name] = serializer.serialize(obj)

        logger.debug(
            'Calling server method "%s", args: %s ...', self.method.name, data
        )
        out = self.call_method(self.name, data, return_raw)
        logger.debug("Server call returned %s", out)
        return self.method.returns.get_serializer().deserialize(out)


class HTTPClient(Client):
    """Access models served with http-based servers"""

    type: ClassVar[str] = "http"
    host: str = "0.0.0.0"
    """Server host"""
    port: Optional[int] = 8080
    """Server port"""

    @property
    def base_url(self):
        prefix = (
            "http://"
            if not self.host.startswith("http://")
            and not self.host.startswith("https://")
            else ""
        )
        if self.port:
            return f"{prefix}{self.host}:{self.port}"
        return f"{prefix}{self.host}"

    def _interface_factory(self) -> InterfaceDescriptor:
        resp = requests.get(f"{self.base_url}/interface.json")
        if resp.status_code != 200:
            try:
                resp.raise_for_status()
            except Exception as e:
                raise MlemError(
                    f"Cannot create client for {self.base_url}"
                ) from e
        return parse_obj_as(VersionedInterfaceDescriptor, resp.json()).methods

    def _call_method(
        self, name: str, args: Any, return_raw: bool
    ):  # pylint: disable=R1710
        ret = requests.post(f"{self.base_url}/{name}", json=args)
        if ret.status_code == 200:  # pylint: disable=R1705
            if return_raw:
                return ret.content
            return ret.json()
        elif ret.status_code == 400:
            raise ExecutionError(ret.json()["error"])
        else:
            ret.raise_for_status()
        return None

    def _call_method_binary(self, name: str, arg: Any, return_raw: bool):
        ret = requests.post(f"{self.base_url}/{name}", files={"file": arg})
        if ret.status_code == 200:  # pylint: disable=R1705
            if return_raw:
                return ret.content  # TODO: change to buffer
            return ret.json()
        elif ret.status_code == 400:
            raise ExecutionError(ret.json()["error"])
        else:
            ret.raise_for_status()
        return None
