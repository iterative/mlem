from typing import ClassVar, Optional, Union
from urllib.parse import urlparse

from pydantic import BaseModel

from mlem.core.objects import (
    DeployState,
    DeployStatus,
    MlemDeployment,
    MlemEnv,
    MlemModel,
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
    """Additional metadata"""


class HerokuEnv(MlemEnv):
    """Heroku Account"""

    type: ClassVar = "heroku"
    api_key: Optional[str] = None
    """HEROKU_API_KEY - advised to set via env variable or `heroku login`"""


class HerokuState(DeployState):
    """State of heroku deployment"""

    type: ClassVar = "heroku"
    app: Optional[HerokuAppMeta]
    """Created heroku app"""
    image: Optional[DockerImage]
    """Built docker image"""
    release_state: Optional[Union[dict, list]]
    """State of the release"""

    @property
    def ensured_app(self) -> HerokuAppMeta:
        if self.app is None:
            raise ValueError("App is not created yet")
        return self.app


class HerokuDeployment(MlemDeployment[HerokuState, HerokuEnv]):
    """Heroku App"""

    type: ClassVar = "heroku"
    state_type: ClassVar = HerokuState
    env_type: ClassVar = HerokuEnv

    app_name: str
    """Heroku application name"""
    region: str = "us"
    """Heroku region"""
    stack: str = "container"
    """Stack to use"""
    team: Optional[str] = None
    """Heroku team"""

    def _get_client(self, state: HerokuState) -> Client:
        return HTTPClient(
            host=urlparse(state.ensured_app.web_url).netloc, port=80
        )

    def deploy(self, model: MlemModel):
        from .utils import create_app, release_docker_app

        with self.lock_state():
            state: HerokuState = self.get_state()
            if state.app is None:
                state.app = create_app(self, api_key=self.get_env().api_key)
                self.update_state(state)

            redeploy = False
            if state.image is None or self.model_changed(model):
                state.image = build_heroku_docker(
                    model, state.app.name, api_key=self.get_env().api_key
                )
                state.update_model(model)
                self.update_state(state)
                redeploy = True
            if state.release_state is None or redeploy:
                state.release_state = release_docker_app(
                    state.app.name,
                    state.image.image_id,
                    api_key=self.get_env().api_key,
                )
                self.update_state(state)

            echo(
                EMOJI_OK
                + f"Service {self.app_name} is up. You can check it out at {state.app.web_url}"
            )

    def remove(self):
        from .utils import delete_app

        with self.lock_state():
            state: HerokuState = self.get_state()

            if state.app is not None:
                delete_app(state.ensured_app.name, self.get_env().api_key)
            self.purge_state()

    def get_status(self, raise_on_error=True) -> DeployStatus:
        from .utils import list_dynos

        state: HerokuState = self.get_state()
        if state.app is None:
            return DeployStatus.NOT_DEPLOYED
        dynos = list_dynos(
            state.ensured_app.name, "web", self.get_env().api_key
        )
        if not dynos:
            if raise_on_error:
                raise DeploymentError(
                    f"No heroku web dynos found, check your dashboard "
                    f"at https://dashboard.heroku.com/apps/{state.ensured_app.name}"
                )
            return DeployStatus.NOT_DEPLOYED
        return HEROKU_STATE_MAPPING[dynos[0]["state"]]
