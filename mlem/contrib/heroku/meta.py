from typing import ClassVar, Optional, Union
from urllib.parse import urlparse

from pydantic import BaseModel

from mlem.core.objects import (
    DeployState,
    DeployStatus,
    MlemDeployment,
    MlemEnv,
)
from mlem.runtime.client import Client, HTTPClient

from ...core.errors import DeploymentError
from ...ui import EMOJI_OK, echo
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
    """App name"""
    web_url: str
    """App web url"""
    meta_info: dict
    """additional metadata"""


class HerokuState(DeployState):
    """State of heroku deployment"""

    type: ClassVar = "heroku"
    app: Optional[HerokuAppMeta]
    """created heroku app"""
    image: Optional[DockerImage]
    """built docker image"""
    release_state: Optional[Union[dict, list]]
    """state of the release"""

    @property
    def ensured_app(self) -> HerokuAppMeta:
        if self.app is None:
            raise ValueError("App is not created yet")
        return self.app


class HerokuDeployment(MlemDeployment):
    """Heroku App"""

    type: ClassVar = "heroku"
    state_type: ClassVar = HerokuState

    app_name: str
    """Heroku application name"""
    region: str = "us"
    """heroku region"""
    stack: str = "container"
    """stack to use"""
    team: Optional[str] = None
    """Heroku team"""

    def _get_client(self, state: HerokuState) -> Client:
        return HTTPClient(
            host=urlparse(state.ensured_app.web_url).netloc, port=80
        )

    def _get_client(self, state: HerokuState) -> Client:
        return HTTPClient(
            host=urlparse(state.ensured_app.web_url).netloc, port=80
        )


class HerokuEnv(MlemEnv[HerokuDeployment]):
    """Heroku Account"""

    type: ClassVar = "heroku"
    deploy_type: ClassVar = HerokuDeployment
    api_key: Optional[str] = None
    """HEROKU_API_KEY - advised to set via env variable or `heroku login`"""

    def deploy(self, meta: HerokuDeployment):
        from .utils import create_app, release_docker_app

        self.check_type(meta)
        with meta.lock_state():
            state: HerokuState = meta.get_state()
            if state.app is None:
                state.app = create_app(meta, api_key=self.api_key)
                meta.update_state(state)

            redeploy = False
            if state.image is None or meta.model_changed():
                state.image = build_heroku_docker(
                    meta.get_model(), state.app.name, api_key=self.api_key
                )
                meta.update_model_hash(state=state)
                redeploy = True
            if state.release_state is None or redeploy:
                state.release_state = release_docker_app(
                    state.app.name,
                    state.image.image_id,
                    api_key=self.api_key,
                )
                meta.update_state(state)

            echo(
                EMOJI_OK
                + f"Service {meta.app_name} is up. You can check it out at {state.app.web_url}"
            )

    def remove(self, meta: HerokuDeployment):
        from .utils import delete_app

        self.check_type(meta)
        with meta.lock_state():
            state: HerokuState = meta.get_state()

            if state.app is not None:
                delete_app(state.ensured_app.name, self.api_key)
            meta.purge_state()

    def get_status(
        self, meta: "HerokuDeployment", raise_on_error=True
    ) -> DeployStatus:
        from .utils import list_dynos

        self.check_type(meta)
        state: HerokuState = meta.get_state()
        if state.app is None:
            return DeployStatus.NOT_DEPLOYED
        dynos = list_dynos(state.ensured_app.name, "web", self.api_key)
        if not dynos:
            if raise_on_error:
                raise DeploymentError(
                    f"No heroku web dynos found, check your dashboard "
                    f"at https://dashboard.heroku.com/apps/{state.ensured_app.name}"
                )
            return DeployStatus.NOT_DEPLOYED
        return HEROKU_STATE_MAPPING[dynos[0]["state"]]
