import os
import posixpath
import shutil
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
from mlem.core.objects import DeployState, DeployStatus, MlemDeployment, \
    MlemEnv, MlemModel
from mlem.runtime.client import Client
from mlem.ui import EMOJI_BUILD, EMOJI_UPLOAD, echo

MODEL_TAR_FILENAME = "model.tar.gz"

class AWSVars(BaseModel):
    bucket: str
    region: str
    account: str
    role_name: str

    @property
    def role(self):
        return f"arn:aws:iam::{self.account}:role/{self.role_name}"

    def get_session(self):
        boto_session = boto3.Session(profile_name=self.account, region_name=self.region)
        return sagemaker.Session(boto_session, default_bucket=self.bucket)

def generate_model_file_name():
    return f"mlem-model-{int(time.time())}"

# class SagemakerDeployment(MlemDeployment):
#     aws_vars: AWSVars
#     image: str
#     model_location: str
#     initial_instance_count: int
#     instance_type: str
#     endpoint_name: str
#     method: str
#     method_descriptor: InterfaceMethodDescriptor
#
#     def get_client(self) -> Client:
#         return SageMakerClient(self)
#
#     # def get_status(self):
#     #     ctx = self.get_predictor().endpoint_context()
#     #     return ctx.properties["Status"]
#
#     def destroy(self):
#         self.get_predictor().delete_endpoint()
#
#     def get_predictor(self):
#         sess = sagemaker.Session(boto3.session.Session())
#         predictor = sagemaker.Predictor(
#             endpoint_name=self.endpoint_name,
#             sagemaker_session=sess,
#             serializer=JSONSerializer(),
#             deserializer=JSONDeserializer(),
#         )
#         return predictor

class SagemakerDeployState(DeployState):
    image: Optional[DockerImage] = None
    model_location: Optional[str] = None
    endpoint_name: Optional[str] = None
    aws_vars: Optional[AWSVars] = None

    def get_predictor(self):
        sess = sagemaker.Session(boto3.session.Session())
        predictor = sagemaker.Predictor(
            endpoint_name=self.endpoint_name,
            sagemaker_session=sess,
            serializer=JSONSerializer(),
            deserializer=JSONDeserializer(),
        )
        return predictor

    def get_client(self):
        pass

class SagemakerDeployment(MlemDeployment):
    type: ClassVar = "sagemaker"
    state_type: ClassVar = SagemakerDeployState

    method: str
    image_name: Optional[str]
    model_file_name: Optional[str] = None
    use_prebuilt: bool = False
    initial_instance_count: int = 1
    instance_type: str = "ml.m4.xlarge"




class SagemakerEnv(MlemEnv):
    deploy_type: ClassVar = SagemakerDeployment

    role_name: Optional[str] = None
    account: Optional[str] = None
    region: Optional[str] = None
    bucket: Optional[str] = None


    @property
    def role(self):
        return f"arn:aws:iam::{self.account}:role/{self.role_name}"

    @staticmethod
    def upload_model(session: sagemaker.Session, model: MlemModel, bucket: str,
                     model_file_name: str):
        with tempfile.TemporaryDirectory() as dirname:
            model.dump(os.path.join(dirname, "model/model"))
            arch_path = os.path.join(dirname, "arch", MODEL_TAR_FILENAME)
            with tarfile.open(
                    arch_path, "w:gz"
            ) as tar:
                path = os.path.join(dirname, "model")
                for file in os.listdir(path):
                    tar.add(os.path.join(path, file), arcname=file)

            model_location = session.upload_data(
                os.path.dirname(arch_path), bucket=bucket, key_prefix=model_file_name
            )

            return model_location


    def deploy(self, meta: SagemakerDeployment):
        with meta.lock_state():
            state: SagemakerDeployState = meta.get_state()
            redeploy = meta.model_changed()
            if not meta.use_prebuilt and (state.image is None or redeploy):
                state.image = build_sagemaker_docker(meta.get_model(), meta.method, self.account, self.region, meta.image_name)
                meta.update_state(state)


            if state.aws_vars is None:
                session, aws_vars = init_aws_vars(self.role_name, self.bucket, self.region, self.account)
                state.aws_vars = aws_vars
                meta.update_state(state)
            else:
                aws_vars = state.aws_vars
                session = aws_vars.get_session()

            if state.model_location is None or redeploy:
                echo(EMOJI_UPLOAD + f"Uploading model distribution to {aws_vars.bucket}...")
                state.model_location = self.upload_model(session, meta.get_model(), aws_vars.bucket,
                                                         meta.model_file_name or generate_model_file_name())
                meta.update_state(state)


            if state.endpoint_name is None or redeploy:
                if state.endpoint_name is None:
                    sm_model = sagemaker.Model(
                        image_uri=state.image.uri,
                        model_data=posixpath.join(state.model_location, MODEL_TAR_FILENAME),
                        role=aws_vars.role,
                        sagemaker_session=session,
                    )
                    echo(
                        EMOJI_BUILD + f"Starting up sagemaker {meta.initial_instance_count} `{meta.instance_type}` instance(s)...")
                    sm_model.deploy(
                        initial_instance_count=meta.initial_instance_count,
                        instance_type=meta.instance_type,
                        wait=False
                    )
                    state.endpoint_name = sm_model.endpoint_name
                    meta.update_state(state)
                else:
                    predictor = state.get_predictor()
                    predictor.update_endpoint(wait=False)


    def remove(self, meta: SagemakerDeployment):
        pass

    def get_status(self, meta: SagemakerDeployment,
                   raise_on_error=True) -> "DeployStatus":
        pass


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


def init_aws_vars(role=None, bucket=None, region=None, account=None):
    boto_session = boto3.Session(profile_name=account, region_name=region)
    sess = sagemaker.Session(boto_session, default_bucket=bucket)

    bucket = (
            bucket or sess.default_bucket()
    )  # Replace with your own bucket name if needed
    region = region or boto_session.region_name
    role = role or sagemaker.get_execution_role(sess)
    account = account or boto_session.client("sts").get_caller_identity().get("Account")
    return sess, AWSVars(bucket=bucket, region=region, account=account, role_name=role)
