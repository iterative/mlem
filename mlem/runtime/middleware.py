from abc import abstractmethod
from typing import ClassVar, List

from pydantic import BaseModel

from mlem.core.base import MlemABC


class Middleware(MlemABC):
    abs_name: ClassVar = "middleware"

    class Config:
        type_root = True

    @abstractmethod
    def on_init(self):
        raise NotImplementedError

    @abstractmethod
    def on_request(self, request):
        raise NotImplementedError

    @abstractmethod
    def on_response(self, request, response):
        raise NotImplementedError


class Middlewares(BaseModel):
    __root__: List[Middleware] = []
    """Middlewares to add to server"""

    def on_init(self):
        for middleware in self.__root__:
            middleware.on_init()

    def on_request(self, request):
        for middleware in self.__root__:
            request = middleware.on_request(request)
        return request

    def on_response(self, request, response):
        for middleware in reversed(self.__root__):
            response = middleware.on_response(request, response)
        return response
