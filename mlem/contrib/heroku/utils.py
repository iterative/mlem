import subprocess
from typing import Optional

import requests
from requests import HTTPError

from ...core.errors import DeploymentError, MlemError
from ...ui import EMOJI_BASE, EMOJI_BUILD, EMOJI_STOP, echo
from .config import HEROKU_CONFIG
from .meta import HerokuAppMeta, HerokuDeployment


def get_api_key() -> str:
    if HEROKU_CONFIG.API_KEY is not None:
        return HEROKU_CONFIG.API_KEY
    try:
        return (
            subprocess.check_output(["heroku", "auth:token"])
            .decode("utf8")
            .strip()
        )
    except subprocess.CalledProcessError as e:
        raise MlemError("HEROKU_API_KEY env is not provided") from e


def heroku_api_request(
    method, route, data=None, accept_version="3", api_key: Optional[str] = None
):
    api_key = api_key or get_api_key()
    if isinstance(method, str):
        method = getattr(requests, method.lower())
    kwargs = {}
    if data is not None:
        kwargs["json"] = data

    r = method(
        "https://api.heroku.com" + route,
        **kwargs,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": f"application/vnd.heroku+json; version={accept_version}",
        },
    )
    try:
        r.raise_for_status()
    except HTTPError as e:
        raise DeploymentError(r.json()["message"]) from e
    return r.json()


def create_app(params: HerokuDeployment, api_key: str = None) -> HerokuAppMeta:
    data = {
        "name": params.app_name,
        "region": params.region,
        "stack": params.stack,
    }
    if params.team is None:
        endpoint = "/apps"
    else:
        data["team"] = params.team
        endpoint = "/teams/apps"

    echo(EMOJI_BASE + f"Creating Heroku App {params.app_name}")
    create_data = heroku_api_request(
        "post",
        endpoint,
        data,
        api_key=api_key,
    )
    res = get_app(params.app_name, params.team, api_key)
    if res is None:
        raise DeploymentError(f"Failed to create {params}: {create_data}")
    return HerokuAppMeta(
        name=res["name"], web_url=res["web_url"], meta_info=res
    )


def get_app(app_name: str, team: str = None, api_key: str = None):
    endpoint = "/apps"
    if team is not None:
        endpoint = "/teams/apps"
    try:
        return heroku_api_request(
            "get", f"{endpoint}/{app_name}", api_key=api_key
        )
    except DeploymentError as e:
        if e.msg == "Couldn't find that app.":
            return None
        raise


def delete_app(app_name: str, api_key: str = None):
    echo(EMOJI_STOP + f"Deleting {app_name} heroku app")
    return heroku_api_request(
        "delete",
        f"/apps/{app_name}",
        api_key=api_key,
    )


def list_dynos(app_name: str, filter_type: str = None, api_key: str = None):
    dynos = heroku_api_request(
        "get",
        f"/apps/{app_name}/dynos",
        api_key=api_key,
    )
    if filter_type is not None:
        dynos = [d for d in dynos if d["type"] == filter_type]
    return dynos


def release_docker_app(
    app_name, image_id, image_type: str = "web", api_key: str = None
):
    echo(EMOJI_BUILD + f"Releasing app {app_name} formation")
    return heroku_api_request(
        "patch",
        f"/apps/{app_name}/formation",
        {"updates": [{"type": image_type, "docker_image": image_id}]},
        accept_version="3.docker-releases",
        api_key=api_key,
    )
