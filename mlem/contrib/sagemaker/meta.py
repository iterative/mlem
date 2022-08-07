import os
import posixpath
import tarfile
import tempfile
from functools import wraps
from typing import ClassVar, Optional, Tuple

import boto3
import sagemaker
from pydantic import validator
from sagemaker.deserializers import JSONDeserializer
from sagemaker.serializers import JSONSerializer

from mlem.config import MlemConfigBase, project_config
from mlem.contrib.docker.base import DockerDaemon, DockerImage
from mlem.contrib.sagemaker.build import (
    AWSVars,
    ECRegistry,
    build_sagemaker_docker,
)
from mlem.core.errors import WrongMethodError
from mlem.core.model import Signature
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
DEFAULT_ECR_REPOSITORY = "mlem"


class AWSConfig(MlemConfigBase):
    ROLE: Optional[str]
    PROFILE: Optional[str]

    class Config:
        section = "aws"
        env_prefix = "AWS_"


def generate_model_file_name(deploy_id):
    return f"mlem-model-{deploy_id}"


def generate_image_name(deploy_id):
    return f"mlem-sagemaker-image-{deploy_id}"


class SagemakerClient(Client):
    endpoint_name: str
    aws_vars: AWSVars
    signature: Signature

    def _interface_factory(self) -> InterfaceDescriptor:
        return InterfaceDescriptor(methods={"predict": self.signature})

    def get_predictor(self):
        sess = self.aws_vars.get_sagemaker_session()
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
    image_tag: Optional[str] = None
    model_location: Optional[str] = None
    endpoint_name: Optional[str] = None
    endpoint_model_hash: Optional[str] = None
    method_signature: Optional[Signature] = None
    region: Optional[str] = None
    previous: Optional["SagemakerDeployState"] = None

    @property
    def image_uri(self):
        if self.image is None:
            if self.image_tag is None:
                raise ValueError(
                    "Cannot get image_uri: image not built or not specified prebuilt image uri"
                )
            return self.image_tag
        return self.image.uri

    def get_predictor(self, session: sagemaker.Session):
        predictor = sagemaker.Predictor(
            endpoint_name=self.endpoint_name,
            sagemaker_session=session,
            serializer=JSONSerializer(),
            deserializer=JSONDeserializer(),
        )
        return predictor


class SagemakerDeployment(MlemDeployment):
    type: ClassVar = "sagemaker"
    state_type: ClassVar = SagemakerDeployState

    method: str = "predict"
    """Model method to be deployed"""
    image_tag: Optional[str] = None
    """Name of the docker image to use"""
    use_prebuilt: bool = False
    """Use pre-built docker image. If True, image_name should be set"""
    model_arch_location: Optional[str] = None
    """Path on s3 to store model archive (excluding bucket)"""
    model_name: Optional[str]
    """Name for SageMaker Model"""
    endpoint_name: Optional[str] = None
    """Name for SageMaker Endpoint"""
    initial_instance_count: int = 1
    """Initial instance count for Endpoint"""
    instance_type: str = "ml.t2.medium"
    """Instance type for Endpoint"""
    accelerator_type: Optional[str] = None
    "The size of the Elastic Inference (EI) instance to use"

    @validator("use_prebuilt")
    def ensure_image_name(  # pylint: disable=no-self-argument
        cls, value, values  # noqa: B902
    ):
        if value and "image_name" not in values:
            raise ValueError(
                "image_name should be set if use_prebuilt is true"
            )
        return value

    def _get_client(self, state: "SagemakerDeployState"):
        return SagemakerClient(
            endpoint_name=state.endpoint_name,
            aws_vars=self.get_env().get_session_and_aws_vars(
                region=state.region
            )[1],
            signature=state.method_signature,
        )


ENDPOINT_STATUS_MAPPING = {
    "Creating": DeployStatus.STARTING,
    "Failed": DeployStatus.CRASHED,
    "InService": DeployStatus.RUNNING,
    "OutOfService": DeployStatus.STOPPED,
    "Updating": DeployStatus.STARTING,
    "SystemUpdating": DeployStatus.STARTING,
    "RollingBack": DeployStatus.STARTING,
    "Deleting": DeployStatus.STOPPED,
}


def updates_state(f):
    @wraps(f)
    def inner(self, meta: MlemDeployment, state: DeployState, *args, **kwargs):
        res = f(self, meta, state, *args, **kwargs)
        meta.update_state(state)
        return res

    return inner


class SagemakerEnv(MlemEnv):
    type: ClassVar = "sagemaker"
    deploy_type: ClassVar = SagemakerDeployment

    role: Optional[str] = None
    account: Optional[str] = None
    region: Optional[str] = None
    bucket: Optional[str] = None
    profile: Optional[str] = None
    ecr_repository: Optional[str] = None

    @property
    def role_name(self):
        return f"arn:aws:iam::{self.account}:role/{self.role}"

    @staticmethod
    def _create_and_upload_model_arch(
        session: sagemaker.Session,
        model: MlemModel,
        bucket: str,
        model_arch_location: str,
    ) -> str:
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
                key_prefix=posixpath.join(
                    model_arch_location, model.meta_hash()
                ),
            )

            return model_location

    @staticmethod
    def _delete_model_file(session: sagemaker.Session, model_path: str):
        s3_client = session.boto_session.client("s3")
        if model_path.startswith("s3://"):
            model_path = model_path[len("s3://") :]
        bucket, *paths = model_path.split("/")
        model_path = posixpath.join(*paths, MODEL_TAR_FILENAME)
        s3_client.delete_object(Bucket=bucket, Key=model_path)

    def deploy(self, meta: SagemakerDeployment):
        with meta.lock_state():
            state: SagemakerDeployState = meta.get_state()
            redeploy = meta.model_changed()
            state.previous = state.previous or SagemakerDeployState()

            session, aws_vars = self.get_session_and_aws_vars(state.region)
            if state.region is None:
                state.region = aws_vars.region
                meta.update_state(state)

            if not meta.use_prebuilt and (state.image_tag is None or redeploy):
                self._build_image(meta, state, aws_vars)

            if state.model_location is None or redeploy:
                self._upload_model(meta, state, aws_vars, session)

            if (
                state.endpoint_name is None
                or redeploy
                or state.endpoint_model_hash is not None
                and state.endpoint_model_hash != state.model_hash
            ):
                if state.endpoint_name is None:
                    self._deploy_model(meta, state, aws_vars, session)
                else:
                    self._update_model(meta, state, aws_vars, session)

    @updates_state
    def _update_model(
        self,
        meta: SagemakerDeployment,
        state: SagemakerDeployState,
        aws_vars: AWSVars,
        session: sagemaker.Session,
    ):
        assert state.model_location is not None  # TODO
        sm_model = sagemaker.Model(
            image_uri=state.image_uri,
            model_data=posixpath.join(
                state.model_location, MODEL_TAR_FILENAME
            ),
            name=meta.model_name,
            role=aws_vars.role,
            sagemaker_session=session,
        )
        sm_model.create(
            instance_type=meta.instance_type,
            accelerator_type=meta.accelerator_type,
        )
        prev_endpoint_conf = session.sagemaker_client.describe_endpoint(
            EndpointName=state.endpoint_name
        )["EndpointConfigName"]
        prev_model_name = session.sagemaker_client.describe_endpoint_config(
            EndpointConfigName=prev_endpoint_conf
        )["ProductionVariants"][0]["ModelName"]

        predictor = state.get_predictor(session)
        predictor.update_endpoint(
            model_name=sm_model.name,
            initial_instance_count=meta.initial_instance_count,
            instance_type=meta.instance_type,
            accelerator_type=meta.accelerator_type,
            wait=True,
        )
        session.sagemaker_client.delete_model(ModelName=prev_model_name)
        prev = state.previous
        if prev is not None:
            if prev.image is not None:
                self._delete_image(meta, prev, aws_vars)
            if prev.model_location is not None:
                self._delete_model_file(session, prev.model_location)
                prev.model_location = None
        session.sagemaker_client.delete_endpoint_config(
            EndpointConfigName=prev_endpoint_conf
        )
        state.endpoint_model_hash = state.model_hash

    @updates_state
    def _delete_image(self, _, state, aws_vars):
        with DockerDaemon(host="").client() as client:
            if isinstance(state.image.registry, ECRegistry):
                state.image.registry.with_aws_vars(aws_vars)
            state.image.delete(client)
            state.image = None

    @updates_state
    def _deploy_model(
        self,
        meta: SagemakerDeployment,
        state: SagemakerDeployState,
        aws_vars: AWSVars,
        session: sagemaker.Session,
    ):
        assert state.model_location is not None  # TODO
        sm_model = sagemaker.Model(
            image_uri=state.image_uri,
            model_data=posixpath.join(
                state.model_location, MODEL_TAR_FILENAME
            ),
            name=meta.model_name,
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
            accelerator_type=meta.accelerator_type,
            endpoint_name=meta.endpoint_name,
            wait=False,
        )
        state.endpoint_name = sm_model.endpoint_name
        state.endpoint_model_hash = state.model_hash

    @updates_state
    def _upload_model(
        self,
        meta: SagemakerDeployment,
        state: SagemakerDeployState,
        aws_vars: AWSVars,
        session: sagemaker.Session,
    ):
        assert state.previous is not None  # TODO
        echo(
            EMOJI_UPLOAD
            + f"Uploading model distribution to {aws_vars.bucket}..."
        )
        if state.model_location is not None:
            state.previous.model_location = state.model_location
        state.model_location = self._create_and_upload_model_arch(
            session,
            meta.get_model(),
            aws_vars.bucket,
            meta.model_arch_location
            or generate_model_file_name(meta.get_model().meta_hash()),
        )
        meta.update_model_hash(state=state)

    @updates_state
    def _build_image(
        self,
        meta: SagemakerDeployment,
        state: SagemakerDeployState,
        aws_vars: AWSVars,
    ):
        assert state.previous is not None  # TODO
        model = meta.get_model()
        try:
            state.method_signature = model.model_type.methods[meta.method]
        except KeyError as e:
            raise WrongMethodError(
                f"Wrong method {meta.method} for model {model.name}"
            ) from e
        image_tag = meta.image_tag or model.meta_hash()
        if state.image_tag is not None:
            state.previous.image_tag = state.image_tag
            state.previous.image = state.image
        state.image = build_sagemaker_docker(
            model,
            meta.method,
            aws_vars.account,
            aws_vars.region,
            image_tag,
            self.ecr_repository or DEFAULT_ECR_REPOSITORY,
            aws_vars,
        )
        state.image_tag = image_tag

    def remove(self, meta: SagemakerDeployment):
        with meta.lock_state():
            state: SagemakerDeployState = meta.get_state()
            session, aws_vars = self.get_session_and_aws_vars(state.region)
            if state.model_location is not None:
                self._delete_model_file(session, state.model_location)
            if state.endpoint_name is not None:

                client = session.sagemaker_client
                endpoint_conf = session.sagemaker_client.describe_endpoint(
                    EndpointName=state.endpoint_name
                )["EndpointConfigName"]

                model_name = client.describe_endpoint_config(
                    EndpointConfigName=endpoint_conf
                )["ProductionVariants"][0]["ModelName"]
                client.delete_model(ModelName=model_name)
                client.delete_endpoint(EndpointName=state.endpoint_name)
                client.delete_endpoint_config(EndpointConfigName=endpoint_conf)
            if state.image is not None:
                self._delete_image(meta, state, aws_vars)
            meta.purge_state()

    def get_status(
        self, meta: SagemakerDeployment, raise_on_error=True
    ) -> "DeployStatus":
        with meta.lock_state():
            state: SagemakerDeployState = meta.get_state()
            session = self.get_session(state.region)

            endpoint = session.sagemaker_client.describe_endpoint(
                EndpointName=state.endpoint_name
            )
            status = endpoint["EndpointStatus"]
            return ENDPOINT_STATUS_MAPPING.get(status, DeployStatus.UNKNOWN)

    def get_session(self, region: str = None) -> sagemaker.Session:
        return self.get_session_and_aws_vars(region)[0]

    def get_session_and_aws_vars(
        self, region: str = None
    ) -> Tuple[sagemaker.Session, AWSVars]:
        return init_aws_vars(
            self.profile,
            self.role,
            self.bucket,
            region or self.region,
            self.account,
        )


def init_aws_vars(
    profile=None, role=None, bucket=None, region=None, account=None
):
    boto_session = boto3.Session(profile_name=profile, region_name=region)
    sess = sagemaker.Session(boto_session, default_bucket=bucket)

    bucket = (
        bucket or sess.default_bucket()
    )  # Replace with your own bucket name if needed
    region = region or boto_session.region_name
    config = project_config(project="", section=AWSConfig)
    role = role or config.ROLE or sagemaker.get_execution_role(sess)
    account = account or boto_session.client("sts").get_caller_identity().get(
        "Account"
    )
    return sess, AWSVars(
        bucket=bucket,
        region=region,
        account=account,
        role_name=role,
        profile=profile or config.PROFILE,
    )
