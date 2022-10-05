import logging
from types import ModuleType
from typing import ClassVar, Dict, List

import boto3
import fastapi
import sagemaker
import uvicorn
from sagemaker.deserializers import JSONDeserializer
from sagemaker.serializers import JSONSerializer

from mlem.config import MlemConfigBase, project_config
from mlem.contrib.fastapi import FastAPIServer
from mlem.contrib.sagemaker.build import AWSVars
from mlem.core.model import Signature
from mlem.runtime import Interface
from mlem.runtime.client import Client
from mlem.runtime.interface import InterfaceDescriptor

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
    """Server to use inside SageMaker containers"""

    type: ClassVar = "_sagemaker"
    libraries: ClassVar[List[ModuleType]] = [
        uvicorn,
        fastapi,
        sagemaker,
        boto3,
    ]
    method: str = local_config.METHOD
    """Method to expose"""
    port: int = local_config.PORT
    """Port to use"""
    host: str = local_config.HOST
    """Host to use"""

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
        return {"SAGEMAKER_METHOD": self.method}


class SagemakerClient(Client):
    """Client to make SageMaker requests"""

    type: ClassVar = "sagemaker"

    endpoint_name: str
    """Name of SageMaker Endpoint"""
    aws_vars: AWSVars
    """AWS Configuration"""
    signature: Signature
    """Signature of deployed method"""

    def _interface_factory(self) -> InterfaceDescriptor:
        return InterfaceDescriptor(methods={"predict": self.signature})

    def get_predictor(self):
        sess = self.aws_vars.get_sagemaker_session()
        predictor = sagemaker.Predictor(
            endpoint_name=self.endpoint_name,
            sagemaker_session=sess,
            serializer=JSONSerializer(),
            deserializer=JSONDeserializer(),
        )
        return predictor

    def _call_method(self, name, args):
        return self.get_predictor().predict(args)
