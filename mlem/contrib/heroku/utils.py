from typing import Optional

import requests
from requests import HTTPError

from ...core.errors import DeploymentError
from .config import HEROKU_CONFIG
from .meta import HerokuAppMeta, HerokuDeploy


def heroku_api_request(
    method, route, data=None, accept_version="3", api_key: Optional[str] = None
):
    api_key = api_key or HEROKU_CONFIG.API_KEY
    if api_key is None:
        raise ValueError("Please set env variable HEROKU_API_KEY")
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


def create_app(params: HerokuDeploy, api_key: str = None) -> HerokuAppMeta:

    res = heroku_api_request(
        "post",
        "/apps",
        {
            "name": params.app_name,
            "region": params.region,
            "stack": params.stack,
        },
        api_key=api_key,
    )
    return HerokuAppMeta(
        name=res["name"], web_url=res["web_url"], meta_info=res
    )


def delete_app(app_name: str, api_key: str = None):
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
    print("Patching formation")
    return heroku_api_request(
        "patch",
        f"/apps/{app_name}/formation",
        {"updates": [{"type": image_type, "docker_image": image_id}]},
        accept_version="3.docker-releases",
        api_key=api_key,
    )
