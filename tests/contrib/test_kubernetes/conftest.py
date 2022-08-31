import os

import pytest
from kubernetes import client, config

from tests.conftest import long


def _cut_empty_lines(string):
    return "\n".join(line for line in string.splitlines() if line)


@pytest.fixture
def k8s_default_manifest():
    return _cut_empty_lines(
        """apiVersion: v1
kind: Namespace
metadata:
  name: mlem-ml-app
  labels:
    name: mlem-ml-app

---

apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml
  namespace: mlem-ml-app
spec:
  selector:
    matchLabels:
      app: ml
  template:
    metadata:
      labels:
        app: ml
    spec:
      containers:
      - name: ml
        image: ml:latest
        imagePullPolicy: Always
        ports:
        - containerPort: 8080

---

apiVersion: v1
kind: Service
metadata:
  name: ml
  namespace: mlem-ml-app
  labels:
    run: ml
spec:
  ports:
  - port: 8080
    protocol: TCP
    targetPort: 8080
  selector:
    app: ml
  type: NodePort
"""
    )


@pytest.fixture
def k8s_manifest():
    return _cut_empty_lines(
        """apiVersion: v1
kind: Namespace
metadata:
  name: mlem-hello-app
  labels:
    name: mlem-hello-app

---

apiVersion: apps/v1
kind: Deployment
metadata:
  name: hello
  namespace: mlem-hello-app
spec:
  selector:
    matchLabels:
      app: hello
  template:
    metadata:
      labels:
        app: hello
    spec:
      containers:
      - name: hello
        image: hello:latest
        imagePullPolicy: Never
        ports:
        - containerPort: 8080

---

apiVersion: v1
kind: Service
metadata:
  name: hello
  namespace: mlem-hello-app
  labels:
    run: hello
spec:
  ports:
  - port: 8080
    protocol: TCP
    targetPort: 8080
  selector:
    app: hello
  type: LoadBalancer
"""
    )


def is_minikube_running() -> bool:
    try:
        config.load_kube_config(
            config_file=os.getenv("KUBECONFIG", default="~/.kube/config")
        )
        client.CoreV1Api().list_namespaced_pod("default")
        return True
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
