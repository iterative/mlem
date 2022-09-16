import pytest

from mlem.contrib.kubernetes.context import (
    ImagePullPolicy,
    K8sYamlBuildArgs,
    K8sYamlGenerator,
)
from mlem.contrib.kubernetes.service import LoadBalancerService
from tests.conftest import _cut_empty_lines


@pytest.fixture
def k8s_default_manifest():
    return _cut_empty_lines(
        """apiVersion: v1
kind: Namespace
metadata:
  name: mlem
  labels:
    name: mlem

---

apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml
  namespace: mlem
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
  namespace: mlem
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
  name: hello
  labels:
    name: hello

---

apiVersion: apps/v1
kind: Deployment
metadata:
  name: test
  namespace: hello
spec:
  selector:
    matchLabels:
      app: test
  template:
    metadata:
      labels:
        app: test
    spec:
      containers:
      - name: test
        image: test:latest
        imagePullPolicy: Never
        ports:
        - containerPort: 8080

---

apiVersion: v1
kind: Service
metadata:
  name: test
  namespace: hello
  labels:
    run: test
spec:
  ports:
  - port: 8080
    protocol: TCP
    targetPort: 8080
  selector:
    app: test
  type: LoadBalancer
"""
    )


def test_k8s_yaml_build_args_default(k8s_default_manifest):
    build_args = K8sYamlBuildArgs()
    assert _generate_k8s_manifest(**build_args.dict()) == k8s_default_manifest


def test_k8s_yaml_build_args(k8s_manifest):
    build_args = K8sYamlBuildArgs(
        namespace="hello",
        image_name="test",
        image_uri="test:latest",
        image_pull_policy=ImagePullPolicy.never,
        port=8080,
        service_type=LoadBalancerService(),
    )
    assert _generate_k8s_manifest(**build_args.dict()) == k8s_manifest


def test_k8s_yaml_generator(k8s_manifest):
    kwargs = {
        "namespace": "hello",
        "image_name": "test",
        "image_uri": "test:latest",
        "image_pull_policy": "Never",
        "port": 8080,
        "service_type": LoadBalancerService(),
    }
    assert _generate_k8s_manifest(**kwargs) == k8s_manifest


def _generate_k8s_manifest(**kwargs):
    return _cut_empty_lines(K8sYamlGenerator(**kwargs).generate())
