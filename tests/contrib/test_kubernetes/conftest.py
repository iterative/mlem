import os

import pytest
from kubernetes import client, config

from tests.conftest import long

from .utils import Command


def is_minikube_running() -> bool:
    try:
        cmd = Command("minikube status")
        returncode = cmd.run(timeout=3, shell=True)
        if returncode == 0:
            config.load_kube_config(
                config_file=os.getenv("KUBECONFIG", default="~/.kube/config")
            )
            client.CoreV1Api().list_namespaced_pod("default")
            return True
        return False
    except (config.config_exception.ConfigException, ConnectionRefusedError):
        return False


def has_k8s():
    if os.environ.get("SKIP_K8S_TESTS", None) == "true":
        return False
    current_os = os.environ.get("GITHUB_MATRIX_OS")
    current_python = os.environ.get("GITHUB_MATRIX_PYTHON")
    if (
        current_os is not None
        and current_os != "ubuntu-latest"
        or current_python is not None
        and current_python != "3.9"
    ):
        return False
    return is_minikube_running()


def k8s_test(f):
    mark = pytest.mark.kubernetes
    skip = pytest.mark.skipif(
        not has_k8s(), reason="kubernetes is unavailable or skipped"
    )
    return long(mark(skip(f)))
