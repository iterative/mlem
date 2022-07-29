import base64
import os

from docker.errors import ImageNotFound

import boto3

from ...core.objects import MlemModel
from ...ui import EMOJI_BUILD, EMOJI_KEY, echo, set_offset
from ..docker.base import DockerEnv, DockerImage, RemoteRegistry
from ..docker.helpers import build_model_image
from .runtime import SageMakerServer

IMAGE_NAME = "mlem-sagemaker-runner"


def ecr_repo_check(image_name, region):
    client = boto3.client("ecr", region_name=region)

    repos = client.describe_repositories()["repositories"]

    if image_name not in {r["repositoryName"] for r in repos}:
        client.create_repository(repositoryName=image_name)



class ECRegistry(RemoteRegistry):
    account: str
    region: str

    def login(self, client):
        ecr = boto3.client("ecr", region_name=self.region)
        auth_data = ecr.get_authorization_token()
        token = auth_data["authorizationData"][0]["authorizationToken"]
        user, token = base64.b64decode(token).decode("utf8").split(":")
        self._login(self.host, client, user, token)
        echo(EMOJI_KEY + f"Logged in to remote registry at host {self.host}")

    @property
    def host(self):
        return f"{self.account}.dkr.ecr.{self.region}.amazonaws.com"

    def image_exists(self, client, image: "DockerImage"):
        ecr = boto3.client("ecr", region_name=self.region)
        images = ecr.list_images(repositoryName=image.name)["imageIds"]
        return len(images) > 0

    def delete_image(
            self, client, image: "DockerImage", force=False, **kwargs
    ):
        if image.image_id is None:
            try:
                docker_image = client.images.get(image.name)
                image_id = docker_image.id
            except ImageNotFound:
                return
        else:
            image_id = image.image_id
        ecr = boto3.client("ecr", region_name=self.region)
        ecr.batch_delete_image(
            repositoryName=image.name,
            imageIds=[{"imageDigest": image_id, "imageTag": image.name}],
        )


def build_sagemaker_docker(
        meta: MlemModel,
        method: str,
        account: str,
        region: str,
        image_name: str,
):
    docker_env = DockerEnv(registry=ECRegistry(account=account, region=region))
    ecr_repo_check(image_name, region)
    echo(EMOJI_BUILD + "Creating docker image for sagemaker")
    with set_offset(2):
        return build_model_image(
        meta,
        image_name,
        server=SageMakerServer(method=method),
        env=docker_env,
        force_overwrite=True,
        templates_dir=os.path.dirname(__file__)
    )
