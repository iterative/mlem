from mlem.contrib.kubernetes.context import K8sYamlBuildArgs, K8sYamlGenerator
from tests.contrib.test_kubernetes.conftest import _cut_empty_lines


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
