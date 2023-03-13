import tempfile
from typing import ClassVar, Optional

from pydantic import BaseModel
from tomlkit import parse

from mlem import ui
from mlem.api import build
from mlem.config import MlemConfigBase, project_config
from mlem.contrib.docker import DockerDirBuilder
from mlem.contrib.flyio.utils import (
    check_flyctl_exec,
    get_scale,
    get_status,
    place_fly_toml,
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
from mlem.runtime.server import Server

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

    image: Optional[str] = None
    """Image name for docker image"""
    app_name: Optional[str] = None
    """Application name. Leave empty for auto-generated one"""
    scale_memory: Optional[int] = None
    """Set VM memory to a number of megabytes (256/512/1024 etc)"""
    # TODO other scale params

    server: Optional[Server] = None
    """Server to use"""

    def _get_client(self, state: FlyioAppState) -> Client:
        return HTTPClient(host=f"https://{state.hostname}", port=443)

    def _create_app(self, state: FlyioAppState):
        with tempfile.TemporaryDirectory(
            prefix="mlem_flyio_build_"
        ) as tempdir:
            args = {
                "auto-confirm": True,
                "region": self.region
                or self.get_env().region
                or project_config("", section=FlyioConfig).region,
                "no-deploy": True,
            }
            if self.app_name:
                args["name"] = self.app_name
            else:
                args["generate-name"] = True
            if self.get_env().access_token:
                args["access-token"] = self.get_env().access_token
            run_flyctl("launch", workdir=tempdir, kwargs=args)
            state.fly_toml = read_fly_toml(tempdir)
            port = getattr(self.server, "port", None) or getattr(
                self.server, "ui_port", None
            )
            if port:  # tell flyio to expose specific port
                fly_toml = parse(state.fly_toml)
                fly_toml["services"][0]["internal_port"] = port
                state.fly_toml = fly_toml.as_string()
            status = get_status(workdir=tempdir)
            state.app_name = status.Name
            state.hostname = status.Hostname
            self.update_state(state)

    def _scale_app(self, state: FlyioAppState):
        if self.scale_memory is None:
            return
        current_scale = get_scale(app_name=state.app_name)

        if current_scale.MemoryMB != self.scale_memory:
            run_flyctl(
                f"scale memory {self.scale_memory}",
                kwargs={"app": state.app_name},
            )
            ui.echo(f"Scaled {state.app_name} memory to {self.scale_memory}MB")

    def _build_in_dir(self, model: MlemModel, state: FlyioAppState):
        with tempfile.TemporaryDirectory(
            prefix="mlem_flyio_build_"
        ) as tempdir:
            assert state.fly_toml is not None
            place_fly_toml(tempdir, state.fly_toml)
            build(DockerDirBuilder(target=tempdir, server=self.server), model)

            args = {}
            if self.get_env().access_token:
                args["access-token"] = self.get_env().access_token

            run_flyctl("deploy", workdir=tempdir, kwargs=args)
            state.fly_toml = read_fly_toml(tempdir)
            state.update_model(model)
            self.update_state(state)
            ui.echo(f"Model deployed to https://{state.hostname}")

    def deploy(self, model: MlemModel):
        check_flyctl_exec()
        with self.lock_state():
            state: FlyioAppState = self.get_state()

            if state.fly_toml is None:
                self._create_app(state)

            self._scale_app(state)

            if self.model_changed(model, state):
                self._build_in_dir(model, state)

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
