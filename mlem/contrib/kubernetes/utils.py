import json
import os
import tempfile

from kubernetes import client, utils, watch

from .context import K8sYamlGenerator


def create_k8s_resources(generator: K8sYamlGenerator):
    k8s_client = client.ApiClient()
    with tempfile.TemporaryDirectory(prefix="mlem_k8s_yaml_build_") as tempdir:
        filename = os.path.join(tempdir, "resource.yaml")
        generator.write(filename)
        try:
            utils.create_from_yaml(k8s_client, filename, verbose=True)
        except utils.FailToCreateError as e:
            failures = e.api_exceptions
            for each_failure in failures:
                error_info = json.loads(each_failure.body)
                if error_info["reason"] != "AlreadyExists":
                    raise e
                if error_info["details"]["kind"] == "deployments":
                    existing_image_uri = (
                        client.CoreV1Api()
                        .list_namespaced_pod(generator.namespace)
                        .items[0]
                        .spec.containers[0]
                        .image
                    )
                    if existing_image_uri != generator.image_uri:
                        api_instance = client.AppsV1Api()
                        body = {
                            "spec": {
                                "template": {
                                    "spec": {
                                        "containers": [
                                            {
                                                "name": generator.image_name,
                                                "image": generator.image_uri,
                                            }
                                        ]
                                    }
                                }
                            }
                        }
                        api_instance.patch_namespaced_deployment(
                            generator.image_name,
                            generator.namespace,
                            body,
                            pretty=True,
                        )


def pod_is_running(namespace, timeout=60) -> bool:
    w = watch.Watch()
    for event in w.stream(
        func=client.CoreV1Api().list_namespaced_pod,
        namespace=namespace,
        timeout_seconds=timeout,
    ):
        if event["object"].status.phase == "Running":
            w.stop()
            return True
    return False


def namespace_deleted(namespace, timeout=60) -> bool:
    w = watch.Watch()
    for event in w.stream(
        func=client.CoreV1Api().list_namespace,
        timeout_seconds=timeout,
    ):
        if (
            namespace == event["object"].metadata.name
            and event["type"] == "DELETED"
        ):
            w.stop()
            return True
    return False
