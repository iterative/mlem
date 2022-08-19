import logging
import os
from typing import ClassVar

from pydantic import BaseModel

from mlem.utils.templates import TemplateModel

logger = logging.getLogger(__name__)


class K8sYamlBuildArgs(BaseModel):
    name: str = "mlem-app"
    image_uri: str = "mlem-app"
    image_pull_policy: str = "Always"
    port: int = 8080
    service_type: str = "NodePort"


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
