from typing import ClassVar, Optional, Union
from urllib.parse import urlparse

from pydantic import BaseModel

from mlem.core.objects import (
    DeployMeta,
    DeployState,
    DeployStatus,
    TargetEnvMeta,
)
from mlem.runtime.client.base import HTTPClient

from ...core.errors import DeploymentError
from ..docker.base import DockerImage
from .build import build_heroku_docker

HEROKU_STATE_MAPPING = {
    "crashed": DeployStatus.CRASHED,
    "down": DeployStatus.STOPPED,
    "idle": DeployStatus.RUNNING,
    "starting": DeployStatus.STARTING,
    "up": DeployStatus.RUNNING,
    "restarting": DeployStatus.STARTING,
}


class HerokuAppMeta(BaseModel):
    name: str
    web_url: str
    meta_info: dict


class HerokuState(DeployState):
    type: ClassVar = "heroku"
    app: Optional[HerokuAppMeta]
    image: Optional[DockerImage]
    release_state: Optional[Union[dict, list]]

    @property
    def ensured_app(self) -> HerokuAppMeta:
        if self.app is None:
            raise ValueError("App is not created yet")
        return self.app

    def get_client(self) -> HTTPClient:
        return HTTPClient(
            host=urlparse(self.ensured_app.web_url).netloc, port=80
        )


class HerokuDeploy(DeployMeta):
    type: ClassVar = "heroku"
    state: Optional[HerokuState]
    app_name: str
    region: str = "us"
    stack: str = "container"
    team: Optional[str] = None


class HerokuEnvMeta(TargetEnvMeta[HerokuDeploy]):
    type: ClassVar = "heroku"
    deploy_type: ClassVar = HerokuDeploy
    api_key: Optional[str] = None

    def deploy(self, meta: HerokuDeploy):
        from .utils import create_app, release_docker_app

        if meta.state is None:
            meta.state = HerokuState()

        meta.update()
        self.check_type(meta)

        if meta.state.app is None:
            meta.state.app = create_app(meta, api_key=self.api_key)
            meta.update()

        if meta.state.image is None:
            meta.state.image = build_heroku_docker(
                meta.get_model(), meta.state.app.name, api_key=self.api_key
            )
            meta.update()
        if meta.state.release_state is None:
            meta.state.release_state = release_docker_app(
                meta.state.app.name,
                meta.state.image.image_id,
                api_key=self.api_key,
            )
            meta.update()

    def destroy(self, meta: HerokuDeploy):
        from .utils import delete_app

        self.check_type(meta)
        if meta.state is None or meta.state.release_state is None:
            return

        delete_app(meta.state.ensured_app.name, self.api_key)
        meta.state = None
        meta.update()

    def get_status(
        self, meta: "HerokuDeploy", raise_on_error=True
    ) -> DeployStatus:
        from .utils import list_dynos

        self.check_type(meta)
        if meta.state is None or meta.state.app is None:
            return DeployStatus.NOT_DEPLOYED
        dynos = list_dynos(meta.state.ensured_app.name, "web", self.api_key)
        if not dynos:
            if raise_on_error:
                raise DeploymentError(
                    f"No heroku web dynos found, check your dashboard "
                    f"at https://dashboard.heroku.com/apps/{meta.state.ensured_app.name}"
                )
            return DeployStatus.NOT_DEPLOYED
        return HEROKU_STATE_MAPPING[dynos[0]["state"]]
