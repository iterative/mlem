"""FastAPI serving
Extension type: serving

FastAPIServer implementation
"""
import logging
from collections.abc import Callable
from types import ModuleType
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type

import fastapi
import uvicorn
from fastapi import FastAPI, UploadFile
from fastapi.datastructures import Default
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, create_model, parse_obj_as
from starlette.responses import JSONResponse, Response, StreamingResponse

from mlem.core.data_type import BinarySerializer, DataSerializer, Serializer
from mlem.core.model import Argument, Signature
from mlem.core.requirements import LibRequirementsMixin
from mlem.runtime.interface import Interface
from mlem.runtime.server import Server
from mlem.ui import EMOJI_NAILS, echo

logger = logging.getLogger(__name__)


def rename_recursively(model: Type[BaseModel], prefix: str):
    model.__name__ = f"{prefix}_{model.__name__}"
    for field in model.__fields__.values():
        if isinstance(field.type_, type) and issubclass(
            field.type_, BaseModel
        ):
            rename_recursively(field.type_, prefix)


def _create_schema_route(app: FastAPI, interface: Interface):
    schema = interface.get_descriptor().dict()
    logger.debug("Creating /interface.json route with schema: %s", schema)
    app.add_api_route("/interface.json", lambda: schema, tags=["schema"])


class FastAPIServer(Server, LibRequirementsMixin):
    """Serves model with http"""

    libraries: ClassVar[List[ModuleType]] = [uvicorn, fastapi]
    type: ClassVar[str] = "fastapi"

    host: str = "0.0.0.0"
    """Network interface to use"""
    port: int = 8080
    """Port to use"""

    @classmethod
    def _create_handler_executor(
        cls,
        method_name: str,
        args: List[Argument],
        arg_serializers: Dict[str, DataSerializer],
        executor: Callable,
        response_serializer: Serializer,
    ):
        deserialized_model = create_model(
            "Model", **{a.name: (Any, ...) for a in args}
        )  # type: ignore[call-overload]

        def serializer_validator(_, values):
            field_values = {}
            for a in args:
                field_values[a.name] = arg_serializers[a.name].deserialize(
                    values[a.name]
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

        if response_serializer.is_binary:
            bin_serializer = response_serializer.binary

            def bin_handler(model: schema_model):  # type: ignore[valid-type]
                values = {a.name: getattr(model, a.name) for a in args}
                result = executor(**values)
                with bin_serializer.dump(result) as buffer:
                    return StreamingResponse(
                        buffer, media_type="application/octet-stream"
                    )

            return bin_handler, None, StreamingResponse

        response_model = response_serializer.data.get_model()
        if issubclass(response_model, BaseModel):
            rename_recursively(response_model, method_name + "_response")

        def handler(model: schema_model):  # type: ignore[valid-type]
            values = {a.name: getattr(model, a.name) for a in args}
            result = executor(**values)
            response = response_serializer.serialize(result)
            return parse_obj_as(response_model, response)

        return handler, response_model, None

    @classmethod
    def _create_handler_executor_binary(
        cls,
        method_name: str,
        serializer: BinarySerializer,
        arg_name: str,
        executor: Callable,
        response_serializer: Serializer,
    ):
        if response_serializer.is_binary:
            bin_serializer = response_serializer.binary

            def bin_handler(file: UploadFile):
                arg = serializer.deserialize(file.file)
                result = executor(**{arg_name: arg})
                with bin_serializer.dump(result) as buffer:
                    return StreamingResponse(
                        buffer, media_type="application/octet-stream"
                    )

            return bin_handler, None, StreamingResponse

        response_model = response_serializer.data.get_model()
        if issubclass(response_model, BaseModel):
            rename_recursively(response_model, method_name + "_response")

        def handler(file: UploadFile):
            arg = serializer.deserialize(file.file)
            result = executor(**{arg_name: arg})

            response = response_serializer.serialize(result)
            return parse_obj_as(response_model, response)

        return handler, response_model, None

    @classmethod
    def _create_handler(
        cls, method_name: str, signature: Signature, executor: Callable
    ) -> Tuple[Optional[Callable], Optional[Type], Optional[Response]]:
        serializers: Dict[str, Serializer] = {
            arg.name: arg.type_.get_serializer() for arg in signature.args
        }
        response_serializer = signature.returns.get_serializer()
        echo(EMOJI_NAILS + f"Adding route for /{method_name}")
        if any(s.is_binary for s in serializers.values()):
            if len(serializers) > 1:
                raise NotImplementedError(
                    "Multiple file requests are not supported yet"
                )
            arg_name = signature.args[0].name
            return cls._create_handler_executor_binary(
                method_name,
                serializers[arg_name].binary,
                arg_name,
                executor,
                response_serializer,
            )
        return cls._create_handler_executor(
            method_name,
            signature.args,
            {k: s.data for k, s in serializers.items()},
            executor,
            response_serializer,
        )

    def app_init(self, interface: Interface):
        app = FastAPI()
        _create_schema_route(app, interface)
        app.add_api_route(
            "/", lambda: RedirectResponse("/docs"), include_in_schema=False
        )

        for method, signature in interface.iter_methods():
            executor = interface.get_method_executor(method)
            handler, response_model, response_class = self._create_handler(
                method, signature, executor
            )

            app.add_api_route(
                f"/{method}",
                handler,
                methods=["POST"],
                response_model=response_model,
                response_class=response_class or Default(JSONResponse),
            )

        return app

    def serve(self, interface: Interface):
        app = self.app_init(interface)
        echo(f"Checkout openapi docs at <http://{self.host}:{self.port}/docs>")
        uvicorn.run(app, host=self.host, port=self.port)
