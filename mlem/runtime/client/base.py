import logging
from abc import ABC, abstractmethod
from typing import Callable, ClassVar, Optional

import requests
from pydantic import BaseModel, parse_obj_as, validator

from mlem.core.base import MlemObject
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
        m_dict = {}
        for method in self.interface.methods:
            m_dict[method.name] = Signature.from_method(method)
        return m_dict

    @abstractmethod
    def _interface_factory(self) -> InterfaceDescriptor:
        pass

    @abstractmethod
    def _call_method(self, name, args):
        pass

    def __getattr__(self, name):
        if name not in self.methods:
            raise KeyError(f"{name} method is not exposed by server")
        return _MethodCall(
            self.base_url, self.methods[name], self._call_method
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
        return parse_obj_as(self.method.returns, out)


class HTTPClient(BaseClient):
    host: str = "localhost"
    port: int = 9000
    base_url: Optional[str] = None

    @validator("base_url", always=True)
    def construct_base_url(self, v, values):
        return v or f'http://{values["host"]}:{values["port"]}'

    def _interface_factory(self) -> InterfaceDescriptor:
        resp = requests.get(f"{self.base_url}/interface.json")
        return parse_obj_as(InterfaceDescriptor, resp.json())

    def _call_method(self, name, args):  # pylint: disable=R1710
        ret = requests.post(f"{self.base_url}/{name}", json=args)
        if ret.status_code == 200:  # pylint: disable=R1705
            return ret.json()["data"]
        elif ret.status_code == 400:
            raise ExecutionError(ret.json()["error"])
        else:
            ret.raise_for_status()
