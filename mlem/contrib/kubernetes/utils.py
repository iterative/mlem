import os
import tempfile

from kubernetes import client, utils, watch


def create_k8s_resources(generator):
    k8s_client = client.ApiClient()
    with tempfile.TemporaryDirectory(prefix="mlem_k8s_yaml_build_") as tempdir:
        filename = os.path.join(tempdir, "resource.yaml")
        generator.write(filename)
        utils.create_from_yaml(k8s_client, filename, verbose=True)


def pod_is_running(namespace, timeout=60) -> bool:
    w = watch.Watch()
    for event in w.stream(
        func=client.CoreV1Api().list_namespaced_pod,
        namespace=namespace,
        timeout_seconds=timeout,
    ):
        print(event["object"].status.phase)
        if event["object"].status.phase == "Running":
            w.stop()
            return True
    return False
