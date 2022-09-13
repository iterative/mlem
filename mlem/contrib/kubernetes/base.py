import os
from typing import ClassVar, List, Optional, Tuple

from kubernetes import client, config
from pydantic import parse_obj_as

from mlem.config import project_config
from mlem.core.base import MlemABC
from mlem.core.errors import DeploymentError, EndpointNotFound, MlemError
from mlem.core.objects import (
    DeployState,
    DeployStatus,
    MlemBuilder,
    MlemDeployment,
    MlemEnv,
    MlemModel,
)
from mlem.runtime.client import Client, HTTPClient
from mlem.runtime.server import Server
from mlem.ui import EMOJI_OK, echo

from ..docker.base import (
    DockerDaemon,
    DockerImage,
    DockerRegistry,
    generate_docker_container_name,
)
from .build import build_k8s_docker
from .context import K8sYamlBuildArgs, K8sYamlGenerator
from .utils import create_k8s_resources, namespace_deleted, pod_is_running

POD_STATE_MAPPING = {
    "Pending": DeployStatus.STARTING,
    "Running": DeployStatus.RUNNING,
    "Succeeded": DeployStatus.STOPPED,
    "Failed": DeployStatus.CRASHED,
    "Unknown": DeployStatus.UNKNOWN,
}


def find_index(nodes_list, node_name):
    for i, each_node in enumerate(nodes_list):
        if each_node.metadata.name == node_name:
            return i
    return -1


class ServiceType(MlemABC):

    abs_name: ClassVar = "k8s_service_type"
    type: ClassVar = "kubernetes"

    class Config:
        type_root = True

    def get_host_and_port(
        self, service, namespace="mlem"  # pylint: disable=unused-argument
    ) -> Tuple[Optional[str], Optional[int]]:
        """Returns host and port for the service in Kubernetes"""
        return None, None


class NodePortService(ServiceType):

    type: ClassVar = "nodeport"

    def get_host_and_port(self, service, namespace="mlem"):
        try:
            port = service.items[0].spec.ports[0].node_port
        except (IndexError, AttributeError) as e:
            raise MlemError(
                "Couldn't determine node port of the deployed service"
            ) from e
        try:
            node_name = (
                client.CoreV1Api()
                .list_namespaced_pod(namespace)
                .items[0]
                .spec.node_name
            )
        except (IndexError, AttributeError) as e:
            raise MlemError(
                "Couldn't determine name of the node where the pod is deployed"
            ) from e
        node_list = client.CoreV1Api().list_node().items
        node_index = find_index(node_list, node_name)
        if node_index == -1:
            raise MlemError(
                f"Couldn't find the node where pods in namespace {namespace} exists"
            )
        address_dict = node_list[node_index].status.addresses
        for each_address in address_dict:
            if each_address.type == "ExternalIP":
                host = each_address.address
                return host, port
        raise EndpointNotFound(
            f"Node {node_name} doesn't have an externally reachable IP address"
        )


class LoadBalancerService(ServiceType):

    type: ClassVar = "loadbalancer"

    def get_host_and_port(self, service, namespace="mlem"):
        try:
            port = service.items[0].spec.ports[0].port
        except (IndexError, AttributeError) as e:
            raise MlemError(
                "Couldn't determine port of the deployed service"
            ) from e
        try:
            ingress = service.items[0].status.load_balancer.ingress[0]
            host = ingress.hostname or ingress.ip
        except (IndexError, AttributeError) as e:
            raise MlemError(
                "Couldn't determine IP address of the deployed service"
            ) from e
        return host, port


class ClusterIPService(ServiceType):

    type: ClassVar = "clusterip"

    def get_host_and_port(self, service, namespace="mlem"):
        raise MlemError(
            "Cannot expose service of type ClusterIP outside the Kubernetes Cluster"
        )


class K8sDeploymentState(DeployState):
    """DeployState implementation for Kubernetes deployments"""

    type: ClassVar = "kubernetes"

    image: Optional[DockerImage] = None
    """Docker Image being used for Deployment"""
    deployment_name: Optional[str] = None
    """Name of Deployment"""


class K8sDeployment(MlemDeployment, K8sYamlBuildArgs):
    """MlemDeployment implementation for Kubernetes deployments"""

    type: ClassVar = "kubernetes"
    state_type: ClassVar = K8sDeploymentState
    """type of state for Kubernetes deployments"""

    server: Optional[Server] = None
    """type of Server to use, with options such as FastAPI, RabbitMQ etc."""
    registry: Optional[DockerRegistry] = DockerRegistry()
    """docker registry"""
    daemon: Optional[DockerDaemon] = DockerDaemon(host="")
    """docker daemon"""
    kube_config_file_path: Optional[str] = None
    """path for kube config file of the cluster"""
    templates_dir: List[str] = []
    """list of dirs where templates reside"""

    @property
    def service_type_spec(self) -> ServiceType:
        return parse_obj_as(ServiceType, {"type": self.service_type.lower()})

    def load_kube_config(self):
        config.load_kube_config(
            config_file=self.kube_config_file_path
            or os.getenv("KUBECONFIG", default="~/.kube/config")
        )

    def _get_client(self, state: K8sDeploymentState) -> Client:
        host, port = None, None
        self.load_kube_config()
        service = client.CoreV1Api().list_namespaced_service(self.namespace)
        try:
            host, port = self.service_type_spec.get_host_and_port(
                service, self.namespace
            )
        except MlemError as e:
            raise EndpointNotFound(
                "Couldn't determine host and port from the service deployed"
            ) from e
        if host is not None and port is not None:
            return HTTPClient(host=host, port=port)
        raise MlemError(
            f"host and port determined are not valid, received host as {host} and port as {port}"
        )


class K8sEnv(MlemEnv[K8sDeployment]):
    """MlemEnv implementation for Kubernetes Environments"""

    type: ClassVar = "kubernetes"
    deploy_type: ClassVar = K8sDeployment
    """type of deployment being used for the Kubernetes environment"""

    registry: Optional[DockerRegistry] = None
    """docker registry"""
    templates_dir: List[str] = []
    """list of dirs where templates reside"""

    def get_registry(self, meta: K8sDeployment):
        registry = meta.registry or self.registry
        if not registry:
            raise MlemError(
                "registry to be used by Docker is not set or supplied"
            )
        return registry

    def get_image_name(self, meta: K8sDeployment):
        return meta.image_name or generate_docker_container_name()

    def get_server(self, meta: K8sDeployment):
        return (
            meta.server
            or project_config(
                meta.loc.project if meta.is_saved else None
            ).server
        )

    def deploy(self, meta: K8sDeployment):
        self.check_type(meta)
        redeploy = False
        with meta.lock_state():
            meta.load_kube_config()
            state: K8sDeploymentState = meta.get_state()
            if state.image is None or meta.model_changed():
                image_name = self.get_image_name(meta)
                state.image = build_k8s_docker(
                    meta=meta.get_model(),
                    image_name=image_name,
                    registry=self.get_registry(meta),
                    daemon=meta.daemon,
                    server=self.get_server(meta),
                )
                meta.update_model_hash(state=state)
                redeploy = True

            if (
                state.deployment_name is None or redeploy
            ) and state.image is not None:
                generator = K8sYamlGenerator(
                    namespace=meta.namespace,
                    image_name=state.image.name,
                    image_uri=state.image.uri,
                    image_pull_policy=meta.image_pull_policy,
                    port=meta.port,
                    service_type=meta.service_type,
                    templates_dir=meta.templates_dir or self.templates_dir,
                )
                create_k8s_resources(generator)

                if pod_is_running(namespace=meta.namespace):
                    deployments_list = (
                        client.AppsV1Api().list_namespaced_deployment(
                            namespace=meta.namespace
                        )
                    )

                    if not len(deployments_list.items):
                        raise DeploymentError(
                            f"Deployment {image_name} couldn't be found in {meta.namespace} namespace"
                        )
                    dpl_name = deployments_list.items[0].metadata.name
                    state.deployment_name = dpl_name
                    meta.update_state(state)

                    echo(
                        EMOJI_OK
                        + f"Deployment {state.deployment_name} is up in {meta.namespace} namespace"
                    )
                else:
                    raise DeploymentError(
                        f"Deployment {image_name} couldn't be set-up on the Kubernetes cluster"
                    )

    def remove(self, meta: K8sDeployment):
        self.check_type(meta)
        with meta.lock_state():
            meta.load_kube_config()
            state: K8sDeploymentState = meta.get_state()
            if state.deployment_name is not None:
                client.CoreV1Api().delete_namespace(name=meta.namespace)
                if namespace_deleted(meta.namespace):
                    echo(
                        EMOJI_OK
                        + f"Deployment {state.deployment_name} and the corresponding service are removed from {meta.namespace} namespace"
                    )
                    state.deployment_name = None
                    meta.update_state(state)

    def get_status(
        self, meta: K8sDeployment, raise_on_error=True
    ) -> DeployStatus:
        self.check_type(meta)
        meta.load_kube_config()
        state: K8sDeploymentState = meta.get_state()
        if state.deployment_name is None:
            return DeployStatus.NOT_DEPLOYED

        pods_list = client.CoreV1Api().list_namespaced_pod(meta.namespace)

        return POD_STATE_MAPPING[pods_list.items[0].status.phase]


class K8sYamlBuilder(MlemBuilder, K8sYamlGenerator):
    """MlemBuilder implementation for building Kubernetes manifests/yamls"""

    type: ClassVar = "kubernetes"

    target: str
    """target path for the manifest/yaml"""

    def build(self, obj: MlemModel):
        self.write(self.target)
        echo(EMOJI_OK + f"{self.target} generated for {obj.basename}")
