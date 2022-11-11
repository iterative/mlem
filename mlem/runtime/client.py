import logging
from abc import ABC, abstractmethod
from typing import Callable, ClassVar, Optional

import requests
from pydantic import BaseModel, parse_obj_as

from mlem.core.base import MlemABC
from mlem.core.errors import MlemError, WrongMethodError
from mlem.core.model import Signature
from mlem.runtime.interface import (
    ExecutionError,
    InterfaceDescriptor,
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

    type: ClassVar[str]
    abs_name: ClassVar[str] = "client"

    @property
    def interface(self):
        return self._interface_factory()

    @property
    def methods(self):
        return self.interface.methods

    @abstractmethod
    def _interface_factory(self) -> InterfaceDescriptor:
        raise NotImplementedError()

    @abstractmethod
    def _call_method(self, name, args):
        raise NotImplementedError()

    def __getattr__(self, name):
        if name not in self.methods:
            raise WrongMethodError(f"{name} method is not exposed by server")
        return _MethodCall(
            method=self.methods[name],
            name=name,
            call_method=self._call_method,
        )


class _MethodCall(BaseModel):
    method: Signature
    name: str
    call_method: Callable

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
        for i, arg in enumerate(self.method.args):
            obj = None
            if len(args) > i:
                obj = args[i]
            if arg.name in kwargs:
                obj = kwargs[arg.name]
            if obj is None:
                raise ValueError(
                    f'Parameter with name "{arg.name}" (position {i}) should be passed'
                )

            data[arg.name] = arg.type_.serialize(obj)

        logger.debug(
            'Calling server method "%s", args: %s ...', self.method.name, data
        )
        out = self.call_method(self.name, data)
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
        return parse_obj_as(VersionedInterfaceDescriptor, resp.json())

    def _call_method(self, name, args):  # pylint: disable=R1710
        ret = requests.post(f"{self.base_url}/{name}", json=args)
        if ret.status_code == 200:  # pylint: disable=R1705
            return ret.json()
        elif ret.status_code == 400:
            raise ExecutionError(ret.json()["error"])
        else:
            ret.raise_for_status()
