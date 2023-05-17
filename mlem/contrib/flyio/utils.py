import json
import logging
import os.path
import subprocess
from typing import Any, Dict

from pydantic import BaseModel, parse_obj_as

from mlem.core.errors import DeploymentError

FLY_TOML = "fly.toml"


logger = logging.getLogger(__name__)


def check_flyctl_exec():
    try:
        cmd = run_flyctl("version --json", wrap_error=False)
        output = json.loads(cmd.decode())
        version = output.get("Version", None)
        if not version or version.split(".")[1] != "1":
            logger.warning(
                "If flyio deployment is causing problems, the `flyctl` version may be the cause. "
                f"You have {version} installed. Try to install `flyctl` 0.1.x."
            )
        return output
    except subprocess.SubprocessError as e:
        raise DeploymentError(
            "flyctl executable is not available. Please install it using <https://fly.io/docs/hands-on/install-flyctl/>"
        ) from e


def run_flyctl(
    command: str,
    workdir: str = None,
    kwargs: Dict[str, Any] = None,
    wrap_error=True,
):
    kwargs = kwargs or {}
    cmd = (
        ["flyctl"]
        + command.split(" ")
        + " ".join(
            [
                f"--{k} {v}" if v is not True else f"--{k}"
                for k, v in kwargs.items()
            ]
        ).split()
    )
    try:
        return subprocess.check_output(cmd, cwd=workdir)
    except subprocess.SubprocessError as e:
        if wrap_error:
            raise DeploymentError(e) from e
        raise


def read_fly_toml(workdir: str):
    with open(os.path.join(workdir, FLY_TOML), encoding="utf8") as f:
        return f.read()


def place_fly_toml(workdir: str, fly_toml: str):
    with open(os.path.join(workdir, FLY_TOML), "w", encoding="utf8") as f:
        f.write(fly_toml)


class FlyioStatusModel(BaseModel):
    Name: str
    Status: str
    Hostname: str


def get_status(workdir: str = None, app_name: str = None) -> FlyioStatusModel:
    args: Dict[str, Any] = {"json": True}
    if app_name is not None:
        args["app"] = app_name
    status = run_flyctl("status", kwargs=args, workdir=workdir)
    return parse_obj_as(FlyioStatusModel, json.loads(status))


class FlyioScaleModel(BaseModel):
    Name: str
    CPUCores: int
    CPUClass: str
    MemoryGB: float
    MemoryMB: int
    PriceMonth: float
    PriceSecond: float
    Count: str
    MaxPerRegion: str


def get_scale(workdir: str = None, app_name: str = None) -> FlyioScaleModel:
    args: Dict[str, Any] = {"json": True}
    if app_name is not None:
        args["app"] = app_name
    status = run_flyctl("scale show", kwargs=args, workdir=workdir)
    return parse_obj_as(FlyioScaleModel, json.loads(status))
