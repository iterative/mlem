import base64
import os
from typing import ClassVar, Optional

import boto3
import sagemaker
from pydantic import BaseModel

from ...core.objects import MlemModel
from ...ui import EMOJI_BUILD, EMOJI_KEY, echo, set_offset
from ..docker.base import DockerEnv, DockerImage, RemoteRegistry
from ..docker.helpers import build_model_image

IMAGE_NAME = "mlem-sagemaker-runner"


class AWSVars(BaseModel):
    """AWS Configuration"""

    profile: str
    """AWS Profile"""
    bucket: str
    """S3 Bucket"""
    region: str
    """AWS Region"""
    account: str
    """AWS Account name"""
    role_name: str
    """AWS Role name"""

    @property
    def role(self):
        return f"arn:aws:iam::{self.account}:role/{self.role_name}"

    def get_sagemaker_session(self):
        return sagemaker.Session(
            self.get_session(), default_bucket=self.bucket
        )

    def get_session(self):
        return boto3.Session(
            profile_name=self.profile, region_name=self.region
        )


def ecr_repo_check(region, repository, session: boto3.Session):
    client = session.client("ecr", region_name=region)

    repos = client.describe_repositories()["repositories"]

    if repository not in {r["repositoryName"] for r in repos}:
        echo(EMOJI_BUILD + f"Creating ECR repository {repository}")
        client.create_repository(repositoryName=repository)


class ECRegistry(RemoteRegistry):
    """ECR registry"""

    class Config:
        exclude = {"aws_vars"}

    type: ClassVar = "ecr"
    account: str
    """AWS Account"""
    region: str
    """AWS Region"""

    aws_vars: Optional[AWSVars] = None
    """AWS Configuration cache"""

    def login(self, client):
        auth_data = self.ecr_client.get_authorization_token()
        token = auth_data["authorizationData"][0]["authorizationToken"]
        user, token = base64.b64decode(token).decode("utf8").split(":")
        self._login(self.get_host(), client, user, token)
        echo(
            EMOJI_KEY
            + f"Logged in to remote registry at host {self.get_host()}"
        )

    def get_host(self) -> Optional[str]:
        return f"{self.account}.dkr.ecr.{self.region}.amazonaws.com"

    def image_exists(self, client, image: DockerImage):
        images = self.ecr_client.list_images(repositoryName=image.name)[
            "imageIds"
        ]
        return len(images) > 0

    def delete_image(self, client, image: DockerImage, force=False, **kwargs):
        return self.ecr_client.batch_delete_image(
            repositoryName=image.name,
            imageIds=[{"imageTag": image.tag}],
        )

    def with_aws_vars(self, aws_vars):
        self.aws_vars = aws_vars
        return self

    @property
    def ecr_client(self):
        return (
            self.aws_vars.get_session().client("ecr")
            if self.aws_vars
            else boto3.client("ecr", region_name=self.region)
        )


def build_sagemaker_docker(
    meta: MlemModel,
    method: str,
    account: str,
    region: str,
    image_name: str,
    repository: str,
    aws_vars: AWSVars,
):
    from .runtime import SageMakerServer  # circular import

    docker_env = DockerEnv(
        registry=ECRegistry(account=account, region=region).with_aws_vars(
            aws_vars
        )
    )
    ecr_repo_check(region, repository, aws_vars.get_session())
    echo(EMOJI_BUILD + "Creating docker image for sagemaker")
    with set_offset(2):
        return build_model_image(
            meta,
            name=repository,
            tag=image_name,
            server=SageMakerServer(method=method),
            env=docker_env,
            force_overwrite=True,
            templates_dir=[os.path.dirname(__file__)],
        )
