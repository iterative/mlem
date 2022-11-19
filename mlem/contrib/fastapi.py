"""FastAPI serving
Extension type: serving

FastAPIServer implementation
"""
import logging
from collections.abc import Callable
from types import ModuleType
from typing import Any, ClassVar, List, Type

import fastapi
import uvicorn
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, create_model, parse_obj_as
from pydantic.typing import get_args

from mlem.core.model import Signature
from mlem.core.requirements import LibRequirementsMixin
from mlem.runtime.interface import Interface
from mlem.runtime.server import Server
from mlem.ui import EMOJI_NAILS, echo

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
    def _create_handler(
        cls, method_name: str, signature: Signature, executor: Callable
    ):
        serializers = {
            arg.name: arg.type_.get_serializer() for arg in signature.args
        }
        kwargs = {
            key: (serializer.get_model(), ...)
            for key, serializer in serializers.items()
        }

        def serializer_validator(_, values):
            field_values = {}
            for a in signature.args:
                field_values[a.name] = serializers[a.name].deserialize(
                    values[a.name]
                )
            return deserialzied_model(**field_values)

        schema_model = create_model(
            "Model",
            **kwargs,
            __validators__={"validate": classmethod(serializer_validator)},
        )  # type: ignore[call-overload]
        rename_recursively(schema_model, method_name + "_request")
        deserialzied_model = create_model(
            "Model",
            **{a.name: (Any, ...) for a in signature.args},
        )  # type: ignore[call-overload]
        response_serializer = signature.returns.get_serializer()
        response_model = response_serializer.get_model()
        if issubclass(response_model, BaseModel):
            rename_recursively(response_model, method_name + "_response")
        echo(EMOJI_NAILS + f"Adding route for /{method_name}")

        def handler(model: schema_model):  # type: ignore[valid-type]
            values = {a.name: getattr(model, a.name) for a in signature.args}
            result = executor(**values)
            response = response_serializer.serialize(result)
            return parse_obj_as(response_model, response)

        return handler, response_model

    def app_init(self, interface: Interface):
        app = FastAPI()
        _create_schema_route(app, interface)
        app.add_api_route(
            "/", lambda: RedirectResponse("/docs"), include_in_schema=False
        )

        for method, signature in interface.iter_methods():
            executor = interface.get_method_executor(method)
            handler, response_model = self._create_handler(
                method, signature, executor
            )

            app.add_api_route(
                f"/{method}",
                handler,
                methods=["POST"],
                response_model=response_model,
            )

        return app

    def serve(self, interface: Interface):
        app = self.app_init(interface)
        echo(f"Checkout openapi docs at <http://{self.host}:{self.port}/docs>")
        uvicorn.run(app, host=self.host, port=self.port)
