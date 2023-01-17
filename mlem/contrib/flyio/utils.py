import json
import os.path
import subprocess
from typing import Any, Dict

from pydantic import BaseModel, parse_obj_as

from mlem.core.errors import DeploymentError

FLY_TOML = "fly.toml"


def check_flyctl_exec():
    try:
        run_flyctl("version", wrap_error=False)
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
