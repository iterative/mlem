import logging
from typing import ClassVar, Dict, Optional

from aiohttp import web
from aiohttp.web_request import Request
from starlette.responses import RedirectResponse

from mlem.config import MlemConfigBase, project_config
from mlem.contrib.fastapi import FastAPIServer
from mlem.core.metadata import load
from mlem.runtime import Interface

logger = logging.getLogger(__name__)



class SageMakerServerConfig(MlemConfigBase):
    HOST: str = "0.0.0.0"
    PORT: int = 8080
    METHOD: str = "predict"

    class Config:
        section = "sagemaker"


local_config = project_config("", section=SageMakerServerConfig)

class SageMakerServer(FastAPIServer):
    type: ClassVar = "sagemaker"
    method: str = local_config.METHOD
    port: int = local_config.PORT
    host: str = local_config.HOST


    def app_init(self, interface: Interface):
        app =  super().app_init(interface)
        app.add_api_route(
            "/invocations", lambda: RedirectResponse(f"/{self.method}")
        )
        app.router.add_get("/ping", lambda request: web.Response(body="OK"))
        return app

    def get_env_vars(self) -> Dict[str, str]:
        return {"SAGAMAKER_METHOD": self.method}


