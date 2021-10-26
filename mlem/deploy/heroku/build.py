import logging
import os

from mlem.contrib.fastapi import FastAPIServer
from mlem.core.objects import ModelMeta
from mlem.core.requirements import MODULE_PACKAGE_MAPPING
from mlem.deploy.heroku.config import HEROKU_API_KEY
from mlem.pack.docker.base import DockerEnv, RemoteRegistry
from mlem.pack.docker.utils import build_image_with_logs
from mlem.runtime import Interface

logger = logging.getLogger(__name__)
MODULE_PACKAGE_MAPPING["yaml"] = "PyYAML"


class HerokuRemoteRegistry(RemoteRegistry):
    def uri(self, image: str):
        uri = super(HerokuRemoteRegistry, self).uri(image)
        return uri.split(":")[0]

    def login(self, client):
        client.login(registry=self.host, username="_", password=HEROKU_API_KEY)


# should we use some abstract server class instead of FastAPI?
# or it's ok for now?
class HerokuServer(FastAPIServer):
    def serve(self, interface: Interface):
        self.port = os.environ.get("PORT")
        logger.info("Switching port to %s", self.port)
        return super().serve(interface)


def build_model_docker(
    meta: ModelMeta, app_name: str, process_type: str = "web"
):
    model = meta.model
    model.wrapper.load(meta.path)
    # this is not needed, it was instead of fsspec to keep artifacts somewhere (crunch)
    # model._unpersisted_artifacts = WrapperArtifactCollection(model.wrapper)
    docker_env = DockerEnv(HerokuRemoteRegistry("registry.heroku.com"))
    return build_image_with_logs(
        process_type,
        model,
        server=HerokuServer(),
        env=docker_env,
        repository=app_name,
        force_overwrite=True,
    )
