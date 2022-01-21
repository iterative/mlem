from typing import Optional

import requests
from requests import HTTPError

from mlem.core.objects import ModelMeta
from mlem.deploy.heroku.config import HEROKU_CONFIG
from mlem.deploy.heroku.meta import HerokuAppMeta


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
    except HTTPError:
        print(r.json())
        raise
    return r.json()


def create_app(
    meta: ModelMeta, name=None, region=None, stack=None, api_key: str = None
) -> HerokuAppMeta:
    name = name or f"mlem-deploy-{meta.name}"
    region = region or "us"
    stack = stack or "container"

    res = heroku_api_request(
        "post",
        "/apps",
        {"name": name, "region": region, "stack": stack},
        api_key=api_key,
    )
    return HerokuAppMeta(
        name=res["name"], web_url=res["web_url"], meta_info=res
    )


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
