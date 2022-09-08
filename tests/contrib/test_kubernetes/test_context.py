import pytest

from mlem.contrib.kubernetes.context import K8sYamlBuildArgs, K8sYamlGenerator
from tests.conftest import _cut_empty_lines


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


def test_k8s_yaml_build_args_default(k8s_default_manifest):
    build_args = K8sYamlBuildArgs()
    assert _generate_k8s_manifest(**build_args.dict()) == k8s_default_manifest


def test_k8s_yaml_build_args(k8s_manifest):
    build_args = K8sYamlBuildArgs(
        image_name="hello",
        image_uri="hello:latest",
        image_pull_policy="Never",
        port=8080,
        service_type="LoadBalancer",
    )
    assert _generate_k8s_manifest(**build_args.dict()) == k8s_manifest


def test_k8s_yaml_generator(k8s_manifest):
    kwargs = {
        "image_name": "hello",
        "image_uri": "hello:latest",
        "image_pull_policy": "Never",
        "port": 8080,
        "service_type": "LoadBalancer",
    }
    assert _generate_k8s_manifest(**kwargs) == k8s_manifest


def _generate_k8s_manifest(**kwargs):
    return _cut_empty_lines(K8sYamlGenerator(**kwargs).generate())
