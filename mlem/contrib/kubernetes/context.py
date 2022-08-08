import logging
import os
from typing import ClassVar, Optional

from pydantic import BaseModel

from mlem.utils.templates import TemplateModel

logger = logging.getLogger(__name__)


class K8sYamlBuildArgs(BaseModel):
    name: Optional[str] = "mlem-app"
    image_uri: Optional[str] = "mlem-app"
    image_pull_policy: Optional[str] = "Always"
    port: Optional[int] = 8080
    service_type: Optional["str"] = "NodePort"


class K8sYamlGenerator(K8sYamlBuildArgs, TemplateModel):
    TEMPLATE_FILE: ClassVar = "resources.yaml.j2"
    TEMPLATE_DIR: ClassVar = os.path.dirname(__file__)

    def prepare_dict(self):
        logger.debug(
            'Generating Resource Yaml via templates from "%s"...',
            self.templates_dir,
        )

        logger.debug('Docker image is based on "%s".', self.image)

        k8s_yaml_args = {
            "name": self.name,
            "image": self.image,
            "image_pull_policy": self.image_pull_policy,
            "port": self.port,
            "service_type": self.service_type,
        }
        return k8s_yaml_args
