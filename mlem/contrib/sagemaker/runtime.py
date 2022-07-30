import logging
from types import ModuleType
from typing import ClassVar, Dict, List

import boto3
import fastapi
import sagemaker
import uvicorn

from mlem.config import MlemConfigBase, project_config
from mlem.contrib.fastapi import FastAPIServer
from mlem.runtime import Interface

logger = logging.getLogger(__name__)


class SageMakerServerConfig(MlemConfigBase):
    HOST: str = "0.0.0.0"
    PORT: int = 8080
    METHOD: str = "predict"

    class Config:
        section = "sagemaker"


local_config = project_config("", section=SageMakerServerConfig)


def ping():
    return "OK"


class SageMakerServer(FastAPIServer):
    type: ClassVar = "sagemaker"
    libraries: ClassVar[List[ModuleType]] = [
        uvicorn,
        fastapi,
        sagemaker,
        boto3,
    ]
    method: str = local_config.METHOD
    port: int = local_config.PORT
    host: str = local_config.HOST

    def app_init(self, interface: Interface):
        app = super().app_init(interface)

        handler, response_model = self._create_handler(
            "invocations",
            interface.get_method_signature(self.method),
            interface.get_method_executor(self.method),
        )
        app.add_api_route(
            "/invocations",
            handler,
            methods=["POST"],
            response_model=response_model,
        )
        app.add_api_route("/ping", ping, methods=["GET"])
        return app

    def get_env_vars(self) -> Dict[str, str]:
        return {"SAGAMAKER_METHOD": self.method}
