import logging
from abc import ABC, abstractmethod
from typing import Callable, ClassVar, Optional

import requests
from pydantic import BaseModel, parse_obj_as

from mlem.core.base import MlemObject
from mlem.core.errors import WrongMethodError
from mlem.core.model import Signature
from mlem.runtime.interface.base import ExecutionError, InterfaceDescriptor

logger = logging.getLogger(__name__)


class BaseClient(MlemObject, ABC):
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
            base_url=self.base_url,
            method=self.methods[name],
            call_method=self._call_method,
        )


class _MethodCall(BaseModel):
    base_url: str
    method: Signature
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
        out = self.call_method(self.method.name, data)
        logger.debug("Server call returned %s", out)
        return self.method.returns.get_serializer().deserialize(out)


class HTTPClient(BaseClient):
    host: str = "0.0.0.0"
    port: Optional[int] = 8080

    @property
    def base_url(self):
        if self.port:
            return f"http://{self.host}:{self.port}"
        return f"http://{self.host}"

    def _interface_factory(self) -> InterfaceDescriptor:
        resp = requests.get(f"{self.base_url}/interface.json")
        return parse_obj_as(InterfaceDescriptor, resp.json())

    def _call_method(self, name, args):  # pylint: disable=R1710
        ret = requests.post(f"{self.base_url}/{name}", json=args)
        if ret.status_code == 200:  # pylint: disable=R1705
            return ret.json()
        elif ret.status_code == 400:
            raise ExecutionError(ret.json()["error"])
        else:
            ret.raise_for_status()
