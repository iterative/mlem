import os
import posixpath
import tarfile
import tempfile
import time
from typing import ClassVar, Optional

import boto3
import sagemaker
from pydantic import BaseModel
from sagemaker.deserializers import JSONDeserializer
from sagemaker.serializers import JSONSerializer

from mlem.contrib.docker.base import DockerImage
from mlem.contrib.sagemaker.build import build_sagemaker_docker
from mlem.core.objects import (
    DeployState,
    DeployStatus,
    MlemDeployment,
    MlemEnv,
    MlemModel,
)
from mlem.runtime.client import Client
from mlem.runtime.interface import InterfaceDescriptor
from mlem.ui import EMOJI_BUILD, EMOJI_UPLOAD, echo

MODEL_TAR_FILENAME = "model.tar.gz"


class AWSVars(BaseModel):
    profile: str
    bucket: str
    region: str
    account: str
    role_name: str

    @property
    def role(self):
        return f"arn:aws:iam::{self.account}:role/{self.role_name}"

    def get_session(self):
        boto_session = boto3.Session(
            profile_name=self.profile, region_name=self.region
        )
        return sagemaker.Session(boto_session, default_bucket=self.bucket)


def generate_model_file_name():
    return f"mlem-model-{int(time.time())}"


def generate_image_name():
    return f"mlem-sagemaker-image-{int(time.time())}"


class SagemakerClient(Client):
    endpoint_name: str
    aws_vars: AWSVars

    def _interface_factory(self) -> InterfaceDescriptor:
        # TMP
        from mlem.core.metadata import load_meta

        model = load_meta("model", force_type=MlemModel)
        sig = model.model_type.methods["predict"]
        return InterfaceDescriptor(methods={"predict": sig})

    def get_predictor(self):
        sess = sagemaker.Session(
            boto3.session.Session(profile_name=self.aws_vars.profile)
        )
        predictor = sagemaker.Predictor(
            endpoint_name=self.endpoint_name,
            sagemaker_session=sess,
            serializer=JSONSerializer(),
            deserializer=JSONDeserializer(),
        )
        return predictor

    def _call_method(self, name, args):
        return self.get_predictor().predict(args)


class SagemakerDeployState(DeployState):
    type: ClassVar = "sagemaker"
    image: Optional[DockerImage] = None
    image_name: Optional[str] = None
    model_location: Optional[str] = None
    endpoint_name: Optional[str] = None
    aws_vars: Optional[AWSVars] = None

    @property
    def image_uri(self):
        if self.image is None:
            if self.image_name is None:
                raise ValueError(
                    "Cannot get image_uri: image not built or not specified prebuilt image uri"
                )
            return self.image_name
        return self.image.uri

    def get_predictor(self):
        sess = sagemaker.Session(
            boto3.session.Session(profile_name=self.aws_vars.profile)
        )
        predictor = sagemaker.Predictor(
            endpoint_name=self.endpoint_name,
            sagemaker_session=sess,
            serializer=JSONSerializer(),
            deserializer=JSONDeserializer(),
        )
        return predictor

    def get_client(self):
        return SagemakerClient(
            endpoint_name=self.endpoint_name, aws_vars=self.aws_vars
        )


class SagemakerDeployment(MlemDeployment):
    type: ClassVar = "sagemaker"
    state_type: ClassVar = SagemakerDeployState

    method: str = "predict"
    image_name: Optional[str] = None
    model_file_name: Optional[str] = None
    use_prebuilt: bool = False
    initial_instance_count: int = 1
    instance_type: str = "ml.m4.xlarge"


ENDPOINT_STATUS_MAPPING = {
    "Creating": DeployStatus.STARTING,
    "Failed": DeployStatus.CRASHED,
    "InService": DeployStatus.RUNNING
    # TODO all statuses
}


class SagemakerEnv(MlemEnv):
    type: ClassVar = "sagemaker"
    deploy_type: ClassVar = SagemakerDeployment

    role: Optional[str] = None
    account: Optional[str] = None
    region: Optional[str] = None
    bucket: Optional[str] = None
    profile: Optional[str] = None

    @property
    def role_name(self):
        return f"arn:aws:iam::{self.account}:role/{self.role}"

    @staticmethod
    def upload_model(
        session: sagemaker.Session,
        model: MlemModel,
        bucket: str,
        model_file_name: str,
    ):
        with tempfile.TemporaryDirectory() as dirname:
            model.clone(os.path.join(dirname, "model", "model"))
            arch_path = os.path.join(dirname, "arch", MODEL_TAR_FILENAME)
            os.makedirs(os.path.dirname(arch_path))
            with tarfile.open(arch_path, "w:gz") as tar:
                path = os.path.join(dirname, "model")
                for file in os.listdir(path):
                    tar.add(os.path.join(path, file), arcname=file)

            model_location = session.upload_data(
                os.path.dirname(arch_path),
                bucket=bucket,
                key_prefix=model_file_name,
            )

            return model_location

    def deploy(self, meta: SagemakerDeployment):
        with meta.lock_state():
            state: SagemakerDeployState = meta.get_state()
            redeploy = meta.model_changed()
            redeploy = False
            if state.aws_vars is None:
                session, aws_vars = init_aws_vars(
                    profile=self.profile,
                    role=self.role,
                    bucket=self.bucket,
                    region=self.region,
                    account=self.account,
                )
                state.aws_vars = aws_vars
                meta.update_state(state)
            else:
                aws_vars = state.aws_vars
                session = aws_vars.get_session()

            if not meta.use_prebuilt and (
                state.image_name is None or redeploy
            ):
                image_name = meta.image_name or generate_image_name()
                state.image = build_sagemaker_docker(
                    meta.get_model(),
                    meta.method,
                    aws_vars.account,
                    aws_vars.region,
                    image_name,
                )
                state.image_name = image_name
                meta.update_state(state)

            if state.model_location is None or redeploy:
                echo(
                    EMOJI_UPLOAD
                    + f"Uploading model distribution to {aws_vars.bucket}..."
                )
                state.model_location = self.upload_model(
                    session,
                    meta.get_model(),
                    aws_vars.bucket,
                    meta.model_file_name or generate_model_file_name(),
                )
                meta.update_model_hash(state=state)
                meta.update_state(state)

            if state.endpoint_name is None or redeploy:
                if state.endpoint_name is None:
                    sm_model = sagemaker.Model(
                        image_uri=state.image_uri,
                        model_data=posixpath.join(
                            state.model_location, MODEL_TAR_FILENAME
                        ),
                        role=aws_vars.role,
                        sagemaker_session=session,
                    )
                    echo(
                        EMOJI_BUILD
                        + f"Starting up sagemaker {meta.initial_instance_count} `{meta.instance_type}` instance(s)..."
                    )
                    sm_model.deploy(
                        initial_instance_count=meta.initial_instance_count,
                        instance_type=meta.instance_type,
                        wait=False,
                    )
                    state.endpoint_name = sm_model.endpoint_name
                    meta.update_state(state)
                else:
                    predictor = state.get_predictor()
                    predictor.update_endpoint(wait=False)

    def remove(self, meta: SagemakerDeployment):
        pass

    def get_status(
        self, meta: SagemakerDeployment, raise_on_error=True
    ) -> "DeployStatus":
        with meta.lock_state():
            state: SagemakerDeployState = meta.get_state()
            aws_vars = state.aws_vars
            session = aws_vars.get_session()
            endpoint = session.sagemaker_client.describe_endpoint(
                EndpointName=state.endpoint_name
            )
            status = endpoint["EndpointStatus"]
            return ENDPOINT_STATUS_MAPPING.get(status, DeployStatus.UNKNOWN)

    #
    # def update(
    #     self, meta: "ModelMeta", previous: "SagemakerDeployment", **kwargs
    # ) -> "Deployment":
    #     from mlem.deploy.sagemaker.command import (
    #         _get_model_method_descriptor,
    #         update_model,
    #     )
    #
    #     method = kwargs["method"]
    #     image = build_model_docker(meta, method, self.account, self.region)
    #     md = _get_model_method_descriptor(meta, method)
    #     update_model(previous, meta.path, method, image.image.uri, md)
    #     return previous


# class SageMakerClient(Client):
#     def __init__(self, meta: SagemakerDeployment):
#         self.meta = meta
#         self.base_url = self.meta.endpoint_name
#         super().__init__()
#
#     def _interface_factory(self) -> InterfaceDescriptor:
#         import mlem
#
#         return InterfaceDescriptor(
#             [self.meta.method_descriptor], mlem.__version__
#         )
#
#     def _call_method(self, name, args):
#         if name != self.meta.method:
#             raise ValueError(f"Wrong method {name}")
#         predictor = self.meta.get_predictor()
#         return predictor.predict(args)["predictions"]


def init_aws_vars(
    profile=None, role=None, bucket=None, region=None, account=None
):
    boto_session = boto3.Session(profile_name=profile, region_name=region)
    sess = sagemaker.Session(boto_session, default_bucket=bucket)

    bucket = (
        bucket or sess.default_bucket()
    )  # Replace with your own bucket name if needed
    region = region or boto_session.region_name
    role = role or sagemaker.get_execution_role(sess)
    account = account or boto_session.client("sts").get_caller_identity().get(
        "Account"
    )
    return sess, AWSVars(
        bucket=bucket,
        region=region,
        account=account,
        role_name=role,
        profile=profile,
    )
