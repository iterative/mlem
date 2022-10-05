import os
import posixpath
import tarfile
import tempfile

import boto3
import sagemaker

from mlem.config import project_config
from mlem.contrib.sagemaker.build import AWSVars
from mlem.contrib.sagemaker.config import AWSConfig
from mlem.core.objects import MlemModel

MODEL_TAR_FILENAME = "model.tar.gz"


def delete_model_file_from_s3(session: sagemaker.Session, model_path: str):
    s3_client = session.boto_session.client("s3")
    if model_path.startswith("s3://"):
        model_path = model_path[len("s3://") :]
    bucket, *paths = model_path.split("/")
    model_path = posixpath.join(*paths, MODEL_TAR_FILENAME)
    s3_client.delete_object(Bucket=bucket, Key=model_path)


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


def _create_model_arch_and_upload_to_s3(
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
            key_prefix=posixpath.join(model_arch_location, model.meta_hash()),
        )

        return model_location


def generate_image_name(deploy_id):
    return f"mlem-sagemaker-image-{deploy_id}"


def generate_model_file_name(deploy_id):
    return f"mlem-model-{deploy_id}"
