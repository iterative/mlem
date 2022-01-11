from dataclasses import dataclass
from pprint import pprint
from urllib.parse import urlparse

from mlem.core.objects import Deployment, ModelMeta, TargetEnvMeta
from mlem.deploy.heroku.build import build_model_docker
from mlem.runtime.client.base import HTTPClient


@dataclass
class HerokuDeployment(Deployment):
    app_name: str
    web_url: str
    meta_info: dict

    def get_client(self) -> HTTPClient:
        return HTTPClient(host=urlparse(self.web_url).netloc, port=80)

    def get_status(self):
        from .utils import heroku_api_request

        dynos = heroku_api_request("get", f"/apps/{self.app_name}/dynos")
        dynos = [d for d in dynos if d["type"] == "web"]
        if not dynos:
            return (
                f"No heroku web dynos found, check your dashboard "
                f"at https://dashboard.heroku.com/apps/{self.app_name}"
            )
        return dynos[0]["state"]

    def destroy(self):
        from .utils import heroku_api_request

        pprint(heroku_api_request("delete", f"/apps/{self.app_name}"))


class HerokuTargetEnvMeta(TargetEnvMeta):
    alias = "heroku"
    deployment_type = HerokuDeployment
    api_key: str = None

    def deploy(self, meta: ModelMeta, **kwargs) -> "Deployment":
        from mlem.deploy.heroku.utils import create_app

        deployment = create_app(meta, api_key=self.api_key)
        return self.update(meta, deployment)

    def update(
        self, meta: ModelMeta, previous: "HerokuDeployment", **kwargs
    ) -> "Deployment":
        from mlem.deploy.heroku.utils import release_docker_app

        image = build_model_docker(meta, previous.app_name)
        release_docker_app(previous.app_name, image.image.image_id, api_key=self.api_key)
        return previous
