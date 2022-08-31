import os
import tempfile
import time

import pytest
from kubernetes import config
from sklearn.datasets import load_iris

from mlem.config import project_config
from mlem.constants import PREDICT_METHOD_NAME
from mlem.contrib.docker.base import DockerDaemon, DockerEnv
from mlem.contrib.docker.helpers import build_model_image
from mlem.contrib.kubernetes.base import (
    K8sDeployment,
    K8sDeploymentState,
    K8sEnv,
)
from mlem.core.objects import DeployStatus

from .utils import Command


@pytest.fixture
def load_kube_config():
    config.load_kube_config(os.getenv("KUBECONFIG", default="~/.kube/config"))


@pytest.fixture
def k8s_deployment(model_meta_saved_single):
    return K8sDeployment(
        name="ml",
        model=model_meta_saved_single.make_link(),
        image_pull_policy="Never",
        service_type="LoadBalancer",
    )


@pytest.fixture
def docker_image(k8s_deployment):
    tmpdir = tempfile.mkdtemp()
    k8s_deployment.dump(os.path.join(tmpdir, "deploy"))
    return build_model_image(
        k8s_deployment.get_model(),
        k8s_deployment.image_name,
        k8s_deployment.server
        or project_config(
            k8s_deployment.loc.project if k8s_deployment.is_saved else None
        ).server,
        DockerEnv(
            daemon=DockerDaemon(host=os.getenv("DOCKER_HOST", default=""))
        ),
        force_overwrite=True,
    )


@pytest.fixture
def k8s_deployment_state(docker_image, model_meta_saved_single):
    return K8sDeploymentState(
        image=docker_image,
        model_hash=model_meta_saved_single.meta_hash(),
    )


@pytest.fixture
def k8s_env():
    return K8sEnv(
        daemon=DockerDaemon(host=os.getenv("DOCKER_HOST", default=""))
    )


def test_deploy(
    load_kube_config, k8s_deployment, k8s_deployment_state, k8s_env
):
    k8s_deployment.update_state(k8s_deployment_state)
    assert k8s_env.get_status(k8s_deployment) == DeployStatus.NOT_DEPLOYED
    k8s_env.deploy(k8s_deployment)
    k8s_deployment.wait_for_status(
        DeployStatus.RUNNING, allowed_intermediate=[DeployStatus.STARTING]
    )
    time.sleep(5)
    assert k8s_env.get_status(k8s_deployment) == DeployStatus.RUNNING
    k8s_env.remove(k8s_deployment)
    time.sleep(5)
    assert k8s_env.get_status(k8s_deployment) == DeployStatus.NOT_DEPLOYED
    with k8s_env.daemon.client() as client:
        k8s_deployment_state.image.delete(client, force=True)


def test_deployed_service(
    load_kube_config, k8s_deployment, k8s_deployment_state, k8s_env
):
    k8s_deployment.update_state(k8s_deployment_state)
    k8s_env.deploy(k8s_deployment)
    cmd = Command("minikube service ml -n mlem-ml-app")
    cmd.run(timeout=5, shell=True)
    client = k8s_deployment.get_client()
    train, _ = load_iris(return_X_y=True)
    response = client.post(
        f"/{PREDICT_METHOD_NAME}",
        json={"data": train},
    )
    assert response.status_code == 200, response.text
    assert response.json() == [0] * 50 + [1] * 50 + [2] * 50
