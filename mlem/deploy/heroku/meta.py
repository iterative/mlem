from pprint import pprint
from typing import Optional
from urllib.parse import urlparse

from pydantic import BaseModel

from mlem.core.objects import DeployMeta, DeployState, TargetEnvMeta
from mlem.deploy.heroku.build import build_model_docker
from mlem.pack.docker.base import DockerImage
from mlem.runtime.client.base import HTTPClient


class HerokuAppMeta(BaseModel):
    name: str
    web_url: str
    meta_info: dict


class HerokuDeployState(DeployState):
    app: Optional[HerokuAppMeta]
    image: Optional[DockerImage]
    release_state: Optional[dict]

    def get_client(self) -> HTTPClient:
        return HTTPClient(host=urlparse(self.app.web_url).netloc, port=80)

    def get_status(self):
        from .utils import heroku_api_request

        dynos = heroku_api_request("get", f"/apps/{self.app.name}/dynos")
        dynos = [d for d in dynos if d["type"] == "web"]
        if not dynos:
            return (
                f"No heroku web dynos found, check your dashboard "
                f"at https://dashboard.heroku.com/apps/{self.app.name}"
            )
        return dynos[0]["state"]

    def destroy(self):
        from .utils import heroku_api_request

        pprint(heroku_api_request("delete", f"/apps/{self.app.name}"))


class HerokuTargetEnvMeta(TargetEnvMeta):
    alias = "heroku"
    deployment_type = HerokuDeployState
    api_key: Optional[str] = None

    def deploy(self, meta: DeployMeta, **kwargs) -> "DeployState":
        from mlem.deploy.heroku.utils import create_app, release_docker_app

        if meta.state is None:
            meta.state = HerokuDeployState()
        if not isinstance(meta.state, HerokuDeployState):
            raise ValueError(
                f"State of the heroku deployment should be {HerokuDeployState}, not {meta.state.__class__}"
            )
        if meta.state.app is None:
            meta.state.app = create_app(meta.get_model(), api_key=self.api_key)
            meta.update()
        if meta.state.image is None:
            meta.state.image = build_model_docker(
                meta.get_model(), meta.state.app.name, api_key=self.api_key
            )
            meta.update()
        if meta.state.release_state is None:
            meta.state.release_state = release_docker_app(
                meta.state.app.name,
                meta.state.image.image_id,
                api_key=self.api_key,
            )
            meta.update()
        return meta.state
