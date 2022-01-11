import logging
import os

from mlem.contrib.fastapi import FastAPIServer
from mlem.core.objects import ModelMeta
from mlem.deploy.heroku.config import HEROKU_CONFIG
from mlem.pack.docker.base import DockerEnv, RemoteRegistry
from mlem.pack.docker.helpers import build_model_image
from mlem.runtime import Interface

logger = logging.getLogger(__name__)


class HerokuRemoteRegistry(RemoteRegistry):
    def uri(self, image: str):
        uri = super(HerokuRemoteRegistry, self).uri(image)
        return uri.split(":")[0]

    def login(self, client):
        client.login(registry=self.host, username="_", password=HEROKU_CONFIG.API_KEY)


class HerokuServer(FastAPIServer):
    def serve(self, interface: Interface):
        self.port = int(os.environ.get("PORT"))
        logger.info("Switching port to %s", self.port)
        return super().serve(interface)


def build_model_docker(
    meta: ModelMeta, app_name: str, process_type: str = "web"
):
    model = meta.model
    docker_env = DockerEnv(registry=HerokuRemoteRegistry(host="registry.heroku.com"))
    return build_model_image(
        model,
        process_type,
        server=HerokuServer(),
        env=docker_env,
        repository=app_name,
        force_overwrite=True,
    )
