import tempfile
from typing import ClassVar, Optional

from pydantic import BaseModel

from mlem.api import build
from mlem.config import MlemConfigBase, project_config
from mlem.contrib.docker import DockerDirBuilder
from mlem.contrib.flyio.utils import (
    check_flyctl_exec,
    get_status,
    read_fly_toml,
    run_flyctl,
)
from mlem.core.objects import (
    DeployState,
    DeployStatus,
    MlemDeployment,
    MlemEnv,
    MlemModel,
)
from mlem.runtime.client import Client, HTTPClient

FLYIO_STATE_MAPPING = {
    "running": DeployStatus.RUNNING,
    "deployed": DeployStatus.RUNNING,
    "pending": DeployStatus.STARTING,
    "suspended": DeployStatus.STOPPED,
    "dead": DeployStatus.CRASHED,
}


class FlyioConfig(MlemConfigBase):
    class Config:
        section = "flyio"

    region: str = "lax"


class FlyioSettings(BaseModel):
    org: Optional[str] = None
    """Organization name"""
    region: Optional[str] = None
    """Region name"""


class FlyioEnv(MlemEnv, FlyioSettings):
    """fly.io organization/account"""

    type: ClassVar = "flyio"

    access_token: Optional[str] = None
    """Access token for fly.io. Alternatively use `flyctl auth login`"""


class FlyioAppState(DeployState):
    """fly.io app state"""

    type: ClassVar = "flyio"

    fly_toml: Optional[str]
    """Contents of fly.toml file for app"""
    app_name: Optional[str]
    """Application name"""
    hostname: Optional[str]
    """Application hostname"""


class FlyioApp(MlemDeployment, FlyioSettings):
    """fly.io deployment"""

    type: ClassVar = "flyio"

    state_type: ClassVar = FlyioAppState
    env_type: ClassVar = FlyioEnv

    image: Optional[str]
    """Image name for docker image"""
    app_name: Optional[str]
    """Application name. Leave empty for auto-generated one"""

    # server: Server = None # TODO
    def _get_client(self, state: FlyioAppState) -> Client:
        return HTTPClient(host=f"https://{state.hostname}", port=443)

    def _build_in_dir(self, model: MlemModel, state: FlyioAppState):
        with tempfile.TemporaryDirectory(
            prefix="mlem_flyio_build_"
        ) as tempdir:
            build(DockerDirBuilder(target=tempdir), model)

            args = {
                "auto-confirm": True,
                "region": self.region
                or self.get_env().region
                or project_config("", section=FlyioConfig).region,
                "now": True,
            }
            if self.app_name:
                args["name"] = self.app_name
            else:
                args["generate-name"] = True
            if self.get_env().access_token:
                args["access-token"] = self.get_env().access_token
            run_flyctl("launch", workdir=tempdir, kwargs=args)
            state.fly_toml = read_fly_toml(tempdir)
            state.update_model(model)

            status = get_status(workdir=tempdir)
            state.app_name = status.Name
            state.hostname = status.Hostname
            self.update_state(state)

    def deploy(self, model: MlemModel):
        check_flyctl_exec()
        with self.lock_state():
            state: FlyioAppState = self.get_state()

            if state.fly_toml is None:
                self._build_in_dir(model, state)
            else:
                raise NotImplementedError("No flyio redeploy yet")

    def remove(self):
        check_flyctl_exec()
        with self.lock_state():
            state: FlyioAppState = self.get_state()
            if state.app_name is not None:
                run_flyctl(
                    f"apps destroy {state.app_name}", kwargs={"yes": True}
                )  # lol
            self.purge_state()

    def get_status(self, raise_on_error=True) -> "DeployStatus":
        check_flyctl_exec()
        with self.lock_state():
            state: FlyioAppState = self.get_state()

            if state.fly_toml is None or state.app_name is None:
                return DeployStatus.NOT_DEPLOYED

            status = get_status(app_name=state.app_name)

            return FLYIO_STATE_MAPPING[status.Status]
