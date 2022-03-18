import logging
import os
from typing import ClassVar, Optional

from mlem.contrib.fastapi import FastAPIServer
from mlem.core.objects import ModelMeta
from mlem.runtime import Interface

from ..docker.base import DockerEnv, DockerImage, RemoteRegistry
from ..docker.helpers import build_model_image
from .config import HEROKU_CONFIG

DEFAULT_HEROKU_REGISTRY = "registry.heroku.com"

logger = logging.getLogger(__name__)


class HerokuRemoteRegistry(RemoteRegistry):
    type: ClassVar = "heroku"
    api_key: Optional[str] = None
    host = DEFAULT_HEROKU_REGISTRY

    def uri(self, image: str):
        return super().uri(image).split(":")[0]

    def login(self, client):
        password = self.api_key or HEROKU_CONFIG.API_KEY
        if password is None:
            raise ValueError(
                "Cannot login to heroku docker registry: no api key provided"
            )
        self._login(self.host, client, "_", password)


class HerokuServer(FastAPIServer):
    type: ClassVar = "heroku"

    def serve(self, interface: Interface):
        self.port = int(os.environ["PORT"])
        logger.info("Switching port to %s", self.port)
        return super().serve(interface)


def build_heroku_docker(
    meta: ModelMeta,
    app_name: str,
    process_type: str = "web",
    api_key: str = None,
    push: bool = True,
) -> DockerImage:
    docker_env = DockerEnv(
        registry=HerokuRemoteRegistry(
            host="registry.heroku.com", api_key=api_key
        )
    )

    return build_model_image(
        meta,
        process_type,
        server=HerokuServer(),
        env=docker_env,
        repository=app_name,
        force_overwrite=True,
        # heroku does not support arm64 images built on Mac M1 devices
        # todo: add this to docs for heroku deploy https://github.com/iterative/mlem/issues/151
        # notice: if you previosly built an arm64 image on the same device,
        # you may cached base images (e.g `python` ) for this image for another architecture and build will fail
        # with message "image with reference sha256:... was found but does not match the specified platform ..."
        platform="linux/amd64",
        push=push,
    )
