import logging
from typing import ClassVar, Optional

from mlem.core.objects import MlemModel

from ...ui import EMOJI_BUILD, echo, set_offset
from ..docker.base import DockerEnv, DockerImage, RemoteRegistry
from ..docker.helpers import build_model_image
from .server import HerokuServer

DEFAULT_HEROKU_REGISTRY = "registry.heroku.com"

logger = logging.getLogger(__name__)


class HerokuRemoteRegistry(RemoteRegistry):
    """Heroku docker registry"""

    type: ClassVar = "heroku"
    api_key: Optional[str] = None
    """HEROKU_API_KEY"""
    host: str = DEFAULT_HEROKU_REGISTRY
    """Registry host"""

    def uri(self, image: str):
        return super().uri(image).split(":")[0]

    def login(self, client):
        from .utils import get_api_key

        password = self.api_key or get_api_key()
        if password is None:
            raise ValueError(
                "Cannot login to heroku docker registry: no api key provided"
            )
        try:
            self._login(self.host, client, "_", password)
        except Exception as e:
            raise ValueError([]) from e


def build_heroku_docker(
    meta: MlemModel,
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
    echo(EMOJI_BUILD + "Creating docker image for heroku")
    with set_offset(2):
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
