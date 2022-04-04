#TODO _bootstrap_method and _MethodCall and Server related changes

from pydantic import validator
import requests
from abc import ABC, abstractmethod
from typing import ClassVar, Dict, Optional
from mlem.core.base import MlemObject
from mlem.runtime.interface.base import ExecutionError, InterfaceDescriptor

class BaseClient(MlemObject, ABC):
    abs_name: ClassVar[str] = "client"
    methods: Dict = {}

    @property
    def interface(self):
        return self._interface_factory()

    @property
    def methods(self):
        for method in self.interface.methods:
            self.methods[method.name] = _bootstrap_method(method)

    @abstractmethod
    def _interface_factory(self) -> InterfaceDescriptor:
        pass

    @abstractmethod
    def _call_method(self, name, args):
        pass

    def __getattr__(self, name):
        if name not in self.methods:
            raise KeyError(f'{name} method is not exposed by server')
        return _MethodCall(self.base_url, self.methods[name], self._call_method)


class HTTPClient(BaseClient):
    host: str = "localhost"
    port: int = 9000
    base_url: Optional[str] = None

    @validator('base_url', always=True)
    def construct_base_url(cls, v, values):
        return v or f'http://{values["host"]}:{values["port"]}'

    def _interface_factory(self) -> InterfaceDescriptor:
        resp = requests.get(f'{self.base_url}/interface.json')
        return InterfaceDescriptor.from_dict(resp.json())

    def _call_method(self, name, args):
        ret = requests.post(f'{self.base_url}/{name}', json=args)
        if ret.status_code == 200:
            return ret.json()['data']
        elif ret.status_code == 400:
            raise ExecutionError(ret.json()['error'])
        else:
            ret.raise_for_status()
