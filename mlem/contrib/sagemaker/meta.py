import posixpath
from functools import wraps
from typing import Any, ClassVar, Optional, Tuple

import sagemaker
from pydantic import validator
from sagemaker.deserializers import JSONDeserializer
from sagemaker.serializers import JSONSerializer
from typing_extensions import Protocol

from mlem.contrib.docker.base import DockerDaemon, DockerImage
from mlem.contrib.sagemaker.build import (
    AWSVars,
    ECRegistry,
    build_sagemaker_docker,
)
from mlem.contrib.sagemaker.runtime import SagemakerClient
from mlem.contrib.sagemaker.utils import (
    MODEL_TAR_FILENAME,
    _create_model_arch_and_upload_to_s3,
    delete_model_file_from_s3,
    generate_model_file_name,
    init_aws_vars,
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
from mlem.ui import EMOJI_BUILD, EMOJI_UPLOAD, echo

DEFAULT_ECR_REPOSITORY = "mlem"


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


class SagemakerDeployState(DeployState):
    """State of SageMaker deployment"""

    type: ClassVar = "sagemaker"

    image: Optional[DockerImage] = None
    """Built image"""
    image_tag: Optional[str] = None
    """Built image tag"""
    model_location: Optional[str] = None
    """Location of uploaded model"""
    endpoint_name: Optional[str] = None
    """Name of SageMaker endpoint"""
    endpoint_model_hash: Optional[str] = None
    """Hash of deployed model"""
    method_signature: Optional[Signature] = None
    """Signature of deployed method"""
    region: Optional[str] = None
    """AWS Region"""
    previous: Optional["SagemakerDeployState"] = None
    """Previous state"""

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


class SagemakerEnv(MlemEnv):
    """SageMaker environment"""

    type: ClassVar = "sagemaker"
    # deploy_type: ClassVar = SagemakerDeployment

    role: Optional[str] = None
    """Default role"""
    account: Optional[str] = None
    """Default account"""
    region: Optional[str] = None
    """Default region"""
    bucket: Optional[str] = None
    """Default bucket"""
    profile: Optional[str] = None
    """Default profile"""
    ecr_repository: Optional[str] = None
    """Default ECR repository"""

    @property
    def role_name(self):
        return f"arn:aws:iam::{self.account}:role/{self.role}"

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


class DeploymentStepMethod(Protocol):
    def __call__(self, state: DeployState, *args, **kwargs) -> Any:
        ...


def updates_state(f) -> DeploymentStepMethod:
    @wraps(f)
    def inner(
        self: MlemDeployment, state: SagemakerDeployState, *args, **kwargs
    ):
        res = f(self, state, *args, **kwargs)
        self.update_state(state)
        return res

    return inner  # type: ignore[return-value]


class SagemakerDeployment(MlemDeployment[SagemakerDeployState, SagemakerEnv]):
    """SageMaker Deployment"""

    type: ClassVar = "sagemaker"
    state_type: ClassVar = SagemakerDeployState
    env_type: ClassVar = SagemakerEnv
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

    @updates_state
    def _upload_model_file(
        self,
        state: SagemakerDeployState,
        model: MlemModel,
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
        state.model_location = _create_model_arch_and_upload_to_s3(
            session,
            model,
            aws_vars.bucket,
            self.model_arch_location
            or generate_model_file_name(model.meta_hash()),
        )
        state.update_model(model)

    @updates_state
    def _update_model(
        self,
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
            name=self.model_name,
            role=aws_vars.role,
            sagemaker_session=session,
        )
        sm_model.create(
            instance_type=self.instance_type,
            accelerator_type=self.accelerator_type,
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
            initial_instance_count=self.initial_instance_count,
            instance_type=self.instance_type,
            accelerator_type=self.accelerator_type,
            wait=True,
        )
        session.sagemaker_client.delete_model(ModelName=prev_model_name)
        prev = state.previous
        if prev is not None:
            if prev.image is not None:
                self._delete_image(prev, aws_vars)
            if prev.model_location is not None:
                delete_model_file_from_s3(session, prev.model_location)
                prev.model_location = None
        session.sagemaker_client.delete_endpoint_config(
            EndpointConfigName=prev_endpoint_conf
        )
        state.endpoint_model_hash = state.model_hash

    @updates_state
    def _build_image(
        self,
        state: SagemakerDeployState,
        model: MlemModel,
        aws_vars: AWSVars,
        ecr_repository: str,
    ):
        assert state.previous is not None  # TODO
        try:
            state.method_signature = model.model_type.methods[self.method]
        except KeyError as e:
            raise WrongMethodError(
                f"Wrong method {self.method} for model {model.name}"
            ) from e
        image_tag = self.image_tag or model.meta_hash()
        if state.image_tag is not None:
            state.previous.image_tag = state.image_tag
            state.previous.image = state.image
        state.image = build_sagemaker_docker(
            model,
            self.method,
            aws_vars.account,
            aws_vars.region,
            image_tag,
            ecr_repository or DEFAULT_ECR_REPOSITORY,
            aws_vars,
        )
        state.image_tag = image_tag

    @updates_state
    def _deploy_model(
        self,
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
            name=self.model_name,
            role=aws_vars.role,
            sagemaker_session=session,
        )
        echo(
            EMOJI_BUILD
            + f"Starting up sagemaker {self.initial_instance_count} `{self.instance_type}` instance(s)..."
        )
        sm_model.deploy(
            initial_instance_count=self.initial_instance_count,
            instance_type=self.instance_type,
            accelerator_type=self.accelerator_type,
            endpoint_name=self.endpoint_name,
            wait=False,
        )
        state.endpoint_name = sm_model.endpoint_name
        state.endpoint_model_hash = state.model_hash

    def deploy(self, model: MlemModel):
        with self.lock_state():
            state: SagemakerDeployState = self.get_state()
            redeploy = self.model_changed(model)
            state.previous = state.previous or SagemakerDeployState()

            session, aws_vars = self.get_env().get_session_and_aws_vars(
                state.region
            )
            if state.region is None:
                state.region = aws_vars.region
                self.update_state(state)

            if not self.use_prebuilt and (state.image_tag is None or redeploy):
                self._build_image(
                    state, model, aws_vars, self.get_env().ecr_repository
                )

            if state.model_location is None or redeploy:
                self._upload_model_file(state, model, aws_vars, session)

            if (
                state.endpoint_name is None
                or redeploy
                or state.endpoint_model_hash is not None
                and state.endpoint_model_hash != state.model_hash
            ):
                if state.endpoint_name is None:
                    self._deploy_model(state, aws_vars, session)
                else:
                    self._update_model(state, aws_vars, session)

    @updates_state
    def _delete_image(self, state: SagemakerDeployState, aws_vars: AWSVars):
        assert state.image is not None  # TODO
        with DockerDaemon(host="").client() as client:
            if isinstance(state.image.registry, ECRegistry):
                state.image.registry.with_aws_vars(aws_vars)
            state.image.delete(client)
            state.image = None

    def remove(self):
        with self.lock_state():
            state: SagemakerDeployState = self.get_state()
            session, aws_vars = self.get_env().get_session_and_aws_vars(
                state.region
            )
            if state.model_location is not None:
                delete_model_file_from_s3(session, state.model_location)
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
                self._delete_image(state, aws_vars)
            self.purge_state()

    def get_status(self, raise_on_error=True) -> "DeployStatus":
        with self.lock_state():
            state: SagemakerDeployState = self.get_state()
            session = self.get_env().get_session(state.region)

            endpoint = session.sagemaker_client.describe_endpoint(
                EndpointName=state.endpoint_name
            )
            status = endpoint["EndpointStatus"]
            return ENDPOINT_STATUS_MAPPING.get(status, DeployStatus.UNKNOWN)
