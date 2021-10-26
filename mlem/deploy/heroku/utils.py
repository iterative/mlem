import requests
from requests import HTTPError

from mlem.core.objects import ModelMeta
from mlem.deploy.heroku.config import HEROKU_API_KEY
from mlem.deploy.heroku.meta import HerokuDeployment


def heroku_api_request(method, route, data=None, accept_version="3"):
    if HEROKU_API_KEY is None:
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
            "Authorization": f"Bearer {HEROKU_API_KEY}",
            "Accept": f"application/vnd.heroku+json; version={accept_version}",
        },
    )
    try:
        r.raise_for_status()
    except HTTPError as e:
        print(r.json())
        raise e
    return r.json()


def create_app(
    meta: ModelMeta, name=None, region=None, stack=None
) -> HerokuDeployment:
    name = name or f'mlem-deploy-{meta.path.replace("/", "-")}'
    region = region or "us"
    stack = stack or "container"

    res = heroku_api_request(
        "post", "/apps", {"name": name, "region": region, "stack": stack}
    )
    return HerokuDeployment(res["name"], res["web_url"], res)


def release_docker_app(app_name, image_id, image_type: str = "web"):
    print("Patching formation")
    return heroku_api_request(
        "patch",
        f"/apps/{app_name}/formation",
        {"updates": [{"type": image_type, "docker_image": image_id}]},
        accept_version="3.docker-releases",
    )
