import logging
import os
from enum import Enum
from typing import ClassVar

from pydantic import BaseModel

from mlem.contrib.kubernetes.service import NodePortService, ServiceType
from mlem.utils.templates import TemplateModel

logger = logging.getLogger(__name__)


class ImagePullPolicy(str, Enum):
    always = "Always"
    never = "Never"
    if_not_present = "IfNotPresent"


class K8sYamlBuildArgs(BaseModel):
    """Class encapsulating parameters for Kubernetes manifests/yamls"""

    class Config:
        use_enum_values = True

    namespace: str = "mlem"
    """Namespace to create kubernetes resources such as pods, service in"""
    image_name: str = "ml"
    """Name of the docker image to be deployed"""
    image_uri: str = "ml:latest"
    """URI of the docker image to be deployed"""
    image_pull_policy: ImagePullPolicy = ImagePullPolicy.always
    """Image pull policy for the docker image to be deployed"""
    port: int = 8080
    """Port where the service should be available"""
    service_type: ServiceType = NodePortService()
    """Type of service by which endpoints of the model are exposed"""


class K8sYamlGenerator(K8sYamlBuildArgs, TemplateModel):
    TEMPLATE_FILE: ClassVar = "resources.yaml.j2"
    TEMPLATE_DIR: ClassVar = os.path.dirname(__file__)

    def prepare_dict(self):
        logger.debug(
            'Generating Resource Yaml via templates from "%s"...',
            self.templates_dir,
        )

        logger.debug('Docker image is based on "%s".', self.image_uri)

        k8s_yaml_args = self.dict()
        k8s_yaml_args["service_type"] = self.service_type.get_string()
        k8s_yaml_args.pop("templates_dir")
        return k8s_yaml_args
