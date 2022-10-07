import os
import re
import subprocess
import tempfile

import numpy as np
import pytest
from kubernetes import config
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

from mlem.api import save
from mlem.config import project_config
from mlem.contrib.docker.base import DockerDaemon, DockerRegistry
from mlem.contrib.kubernetes.base import K8sDeployment, K8sDeploymentState
from mlem.contrib.kubernetes.build import build_k8s_docker
from mlem.contrib.kubernetes.context import ImagePullPolicy
from mlem.contrib.kubernetes.service import LoadBalancerService
from mlem.core.objects import DeployStatus, MlemModel
from tests.contrib.test_kubernetes.conftest import k8s_test
from tests.contrib.test_kubernetes.utils import Command


@pytest.fixture(scope="session")
def minikube_env_variables():
    old_environ = dict(os.environ)
    output = subprocess.check_output(
        ["minikube", "-p", "minikube", "docker-env"]
    )
    export_re = re.compile('export ([A-Z_]+)="(.*)"\\n')
    export_pairs = export_re.findall(output.decode("UTF-8"))
    for k, v in export_pairs:
        os.environ[k] = v

    yield

    os.environ.clear()
    os.environ.update(old_environ)


@pytest.fixture
def load_kube_config():
    config.load_kube_config(os.getenv("KUBECONFIG", default="~/.kube/config"))


@pytest.fixture(scope="session")
def model_meta(tmp_path_factory):
    path = os.path.join(tmp_path_factory.getbasetemp(), "saved-model-single")
    train, target = load_iris(return_X_y=True)
    model = DecisionTreeClassifier().fit(train, target)
    return save(model, path, sample_data=train)


@pytest.fixture(scope="session")
def k8s_deployment(minikube_env_variables):
    return K8sDeployment(
        namespace="ml",
        image_pull_policy=ImagePullPolicy.never,
        service_type=LoadBalancerService(),
        daemon=DockerDaemon(host=os.getenv("DOCKER_HOST", default="")),
    )


@pytest.fixture(scope="session")
def docker_image(k8s_deployment, model_meta):
    tmpdir = tempfile.mkdtemp()
    k8s_deployment.dump(os.path.join(tmpdir, "deploy"))
    return build_k8s_docker(
        model_meta,
        k8s_deployment.image_name,
        DockerRegistry(),
        DockerDaemon(host=os.getenv("DOCKER_HOST", default="")),
        k8s_deployment.server or project_config(None).server,
        platform=None,
    )


@pytest.fixture
def k8s_deployment_state(docker_image, model_meta, k8s_deployment):
    return K8sDeploymentState(
        image=docker_image,
        model_hash=model_meta.meta_hash(),
        declaration=k8s_deployment,
    )


@k8s_test
@pytest.mark.usefixtures("load_kube_config")
def test_deploy(
    k8s_deployment: K8sDeployment,
    k8s_deployment_state: K8sDeploymentState,
    model_meta: MlemModel,
):
    k8s_deployment.update_state(k8s_deployment_state)
    assert (
        k8s_deployment.get_status(k8s_deployment) == DeployStatus.NOT_DEPLOYED
    )
    k8s_deployment.deploy(model_meta)
    k8s_deployment.wait_for_status(
        DeployStatus.RUNNING,
        allowed_intermediate=[DeployStatus.STARTING],
        timeout=10,
        times=5,
    )
    assert k8s_deployment.get_status(k8s_deployment) == DeployStatus.RUNNING
    k8s_deployment.remove()
    assert (
        k8s_deployment.get_status(k8s_deployment) == DeployStatus.NOT_DEPLOYED
    )


@k8s_test
@pytest.mark.usefixtures("load_kube_config")
def test_deployed_service(
    k8s_deployment: K8sDeployment,
    k8s_deployment_state: K8sDeploymentState,
    model_meta: MlemModel,
):
    k8s_deployment.update_state(k8s_deployment_state)
    k8s_deployment.deploy(model_meta)
    cmd = Command("minikube tunnel")
    cmd.run(timeout=20, shell=True)
    client = k8s_deployment.get_client()
    train, _ = load_iris(return_X_y=True)
    response = client.predict(data=train)
    assert np.array_equal(response, np.array([0] * 50 + [1] * 50 + [2] * 50))
