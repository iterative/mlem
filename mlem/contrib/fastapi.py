import logging
from collections.abc import Callable
from types import ModuleType
from typing import ClassVar, List, Type

import fastapi
import uvicorn
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, create_model, parse_obj_as

from mlem.core.model import Signature
from mlem.core.requirements import LibRequirementsMixin
from mlem.runtime.interface.base import Interface
from mlem.runtime.server.base import Server
from mlem.ui import EMOJI_NAILS, echo

logger = logging.getLogger(__name__)


def rename_recursively(model: Type[BaseModel], prefix: str):
    model.__name__ = f"{prefix}_{model.__name__}"
    for field in model.__fields__.values():
        if issubclass(field.type_, BaseModel):
            rename_recursively(field.type_, prefix)


def _create_schema_route(app: FastAPI, interface: Interface):
    schema = interface.get_descriptor().dict()
    logger.debug("Creating /interface.json route with schema: %s", schema)
    app.add_api_route("/interface.json", lambda: schema, tags=["schema"])


class FastAPIServer(Server, LibRequirementsMixin):
    libraries: ClassVar[List[ModuleType]] = [uvicorn, fastapi]
    type: ClassVar[str] = "fastapi"

    host: str = "0.0.0.0"
    port: int = 8080

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
        payload_model = create_model("Model", **kwargs)  # type: ignore
        rename_recursively(payload_model, method_name + "_request")
        response_serializer = signature.returns.get_serializer()
        response_model = response_serializer.get_model()
        rename_recursively(response_model, method_name + "_response")
        echo(EMOJI_NAILS + f"Adding route for /{method_name}")

        def handler(model: payload_model):  # type: ignore[valid-type]
            kwargs = {}
            # TODO: https://github.com/iterative/mlem/issues/149
            for a in signature.args:
                d = getattr(model, a.name).dict()
                obj = d.get("__root__", None)
                if obj is not None:
                    kwargs[a.name] = serializers[a.name].deserialize(obj)
                else:
                    kwargs[a.name] = serializers[a.name].deserialize(d)
            result = executor(**kwargs)
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
