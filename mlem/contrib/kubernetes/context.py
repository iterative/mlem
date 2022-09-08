import logging
import os
from enum import Enum
from typing import ClassVar

from pydantic import BaseModel

from mlem.utils.templates import TemplateModel

logger = logging.getLogger(__name__)


class ImagePullPolicy(str, Enum):
    always = "Always"
    never = "Never"
    if_not_present = "IfNotPresent"


class ServiceType(str, Enum):
    cluster_ip = "ClusterIP"
    node_port = "NodePort"
    load_balancer = "LoadBalancer"


class K8sYamlBuildArgs(BaseModel):
    """class encapsulating parameters for Kubernetes manifests/yamls"""

    class Config:
        use_enum_values = True

    namespace: str = "mlem"
    image_name: str = "ml"
    image_uri: str = "ml:latest"
    image_pull_policy: ImagePullPolicy = ImagePullPolicy.always
    port: int = 8080
    service_type: ServiceType = ServiceType.node_port


class K8sYamlGenerator(K8sYamlBuildArgs, TemplateModel):
    TEMPLATE_FILE: ClassVar = "resources.yaml.j2"
    TEMPLATE_DIR: ClassVar = os.path.dirname(__file__)

    def prepare_dict(self):
        logger.debug(
            'Generating Resource Yaml via templates from "%s"...',
            self.templates_dir,
        )

        logger.debug('Docker image is based on "%s".', self.image_uri)

        k8s_yaml_args = self.dict().copy()
        k8s_yaml_args.pop("templates_dir")
        return k8s_yaml_args
