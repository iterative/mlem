"""FastAPI serving
Extension type: serving

FastAPIServer implementation
"""
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from types import ModuleType
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type

import fastapi
import uvicorn
from fastapi import FastAPI, Request, UploadFile
from fastapi.datastructures import Default
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, create_model, parse_obj_as
from pydantic.typing import get_args
from starlette.responses import JSONResponse, Response, StreamingResponse

from mlem.core.data_type import DataTypeSerializer
from mlem.core.requirements import LibRequirementsMixin, Requirements
from mlem.runtime.interface import (
    Interface,
    InterfaceArgument,
    InterfaceMethod,
)
from mlem.runtime.middleware import Middleware, Middlewares
from mlem.runtime.server import Server
from mlem.ui import EMOJI_NAILS, echo
from mlem.utils.module import get_object_requirements

logger = logging.getLogger(__name__)


def rename_recursively(model: Type, prefix: str):
    if isinstance(model, type):
        if not issubclass(model, BaseModel):
            return
        model.__name__ = f"{prefix}_{model.__name__}"
        for field in model.__fields__.values():
            rename_recursively(field.type_, prefix)
    for arg in get_args(model):
        rename_recursively(arg, prefix)


def _create_schema_route(app: FastAPI, interface: Interface):
    schema = interface.get_versioned_descriptor().dict()
    logger.debug("Creating /interface.json route with schema: %s", schema)
    app.add_api_route("/interface.json", lambda: schema, tags=["schema"])


class FastAPIMiddleware(Middleware, ABC):
    @abstractmethod
    def on_app_init(self, app: FastAPI):
        raise NotImplementedError


class FastAPIServer(Server, LibRequirementsMixin):
    """Serves model with http"""

    libraries: ClassVar[List[ModuleType]] = [uvicorn, fastapi]
    type: ClassVar[str] = "fastapi"
    port_field: ClassVar = "port"

    host: str = "0.0.0.0"
    """Network interface to use"""
    port: int = 8080
    """Port to use"""
    debug: bool = False
    """If true will not wrap exceptions"""

    @classmethod
    def _create_handler_executor(
        cls,
        method_name: str,
        args: Dict[str, InterfaceArgument],
        arg_serializers: Dict[str, DataTypeSerializer],
        executor: Callable,
        response_serializer: DataTypeSerializer,
        middlewares: Middlewares,
    ):
        deserialized_model = create_model(
            "Model", **{a: (Any, ...) for a in args}
        )  # type: ignore[call-overload]

        def serializer_validator(_, values):
            field_values = {}
            for name in args:
                field_values[name] = arg_serializers[name].deserialize(
                    values[name]
                )
            return deserialized_model(**field_values)

        arg_models = {
            key: (serializer.get_model(), ...)
            for key, serializer in arg_serializers.items()
        }

        schema_model = create_model(
            "Model",
            **arg_models,
            __validators__={"validate": classmethod(serializer_validator)},
        )  # type: ignore[call-overload]
        rename_recursively(schema_model, method_name + "_request")

        if response_serializer.serializer.is_binary:

            def bin_handler(model: schema_model):  # type: ignore[valid-type]
                values = {a: getattr(model, a) for a in args}
                values = middlewares.on_request(values)
                result = executor(**values)
                result = middlewares.on_response(values, result)
                with response_serializer.dump(result) as buffer:
                    return StreamingResponse(
                        buffer, media_type="application/octet-stream"
                    )

            return bin_handler, None, StreamingResponse

        response_model = response_serializer.get_model()
        if issubclass(response_model, BaseModel):
            rename_recursively(response_model, method_name + "_response")

        def handler(model: schema_model):  # type: ignore[valid-type]
            values = {a: getattr(model, a) for a in args}
            values = middlewares.on_request(values)
            result = executor(**values)
            result = middlewares.on_response(values, result)
            response = response_serializer.serialize(result)
            return parse_obj_as(response_model, response)

        return handler, response_model, None

    @classmethod
    def _create_handler_executor_binary(
        cls,
        method_name: str,
        serializer: DataTypeSerializer,
        arg_name: str,
        executor: Callable,
        response_serializer: DataTypeSerializer,
        middlewares: Middlewares,
    ):
        if response_serializer.serializer.is_binary:

            def bin_handler(file: UploadFile):
                arg = serializer.deserialize(_SpooledFileIOWrapper(file.file))
                arg = middlewares.on_request(arg)
                result = executor(**{arg_name: arg})
                result = middlewares.on_response(arg, result)
                with response_serializer.dump(result) as buffer:
                    return StreamingResponse(
                        buffer, media_type="application/octet-stream"
                    )

            return bin_handler, None, StreamingResponse

        response_model = response_serializer.get_model()
        if issubclass(response_model, BaseModel):
            rename_recursively(response_model, method_name + "_response")

        def handler(file: UploadFile):
            arg = serializer.deserialize(file.file)
            arg = middlewares.on_request(arg)
            result = executor(**{arg_name: arg})
            result = middlewares.on_response(arg, result)
            response = response_serializer.serialize(result)
            return parse_obj_as(response_model, response)

        return handler, response_model, None

    def _create_handler(
        self,
        method_name: str,
        signature: InterfaceMethod,
        executor: Callable,
        middlewares: Middlewares,
    ) -> Tuple[Optional[Callable], Optional[Type], Optional[Response]]:
        serializers, response_serializer = self._get_serializers(signature)
        echo(EMOJI_NAILS + f"Adding route for /{method_name}")
        if any(s.serializer.is_binary for s in serializers.values()):
            if len(serializers) > 1:
                raise NotImplementedError(
                    "Multiple file requests are not supported yet"
                )
            arg_name = signature.args[0].name
            return self._create_handler_executor_binary(
                method_name,
                serializers[arg_name],
                arg_name,
                executor,
                response_serializer,
                middlewares,
            )
        return self._create_handler_executor(
            method_name,
            {a.name: a for a in signature.args},
            serializers,
            executor,
            response_serializer,
            middlewares,
        )

    def app_init(self, interface: Interface):
        app = FastAPI()
        _create_schema_route(app, interface)
        app.add_api_route(
            "/", lambda: RedirectResponse("/docs"), include_in_schema=False
        )
        for mid in self.middlewares.__root__:
            mid.on_init()
            if isinstance(mid, FastAPIMiddleware):
                mid.on_app_init(app)

        for method, signature in interface.iter_methods():
            executor = interface.get_method_executor(method)
            handler, response_model, response_class = self._create_handler(
                method, signature, executor, self.middlewares
            )

            app.add_api_route(
                f"/{method}",
                handler,
                methods=["POST"],
                response_model=response_model,
                response_class=response_class or Default(JSONResponse),
            )

        if not self.debug:
            # pylint: disable=unused-argument
            @app.exception_handler(Exception)
            def exception_handler(request: Request, exc: Exception):
                return JSONResponse(
                    status_code=400,
                    content={"error": f"{exc.__class__.__name__}: {exc}"},
                )

        return app

    def serve(self, interface: Interface):
        app = self.app_init(interface)
        echo(f"Checkout openapi docs at <http://{self.host}:{self.port}/docs>")
        uvicorn.run(app, host=self.host, port=self.port)

    def get_requirements(self) -> Requirements:
        return super().get_requirements() + get_object_requirements(
            [self.request_serializer, self.response_serializer, self.methods]
        )


class _SpooledFileIOWrapper:
    """https://stackoverflow.com/questions/47160211/why-doesnt-tempfile-spooledtemporaryfile-implement-readable-writable-seekable
    Waiting for 3.10 EOL to drop this
    """

    def __init__(self, _file):
        self.__file = _file

    def __getattr__(self, item):
        return getattr(self.__file, item)

    @property
    def readable(self):
        return self.__file._file.readable  # pylint: disable=protected-access

    @property
    def writable(self):
        return self.__file._file.writable  # pylint: disable=protected-access

    @property
    def seekable(self):
        return self.__file._file.seekable  # pylint: disable=protected-access
