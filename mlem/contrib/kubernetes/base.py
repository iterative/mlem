import os
from typing import ClassVar, List, Optional

from kubernetes import client, config

from mlem.config import project_config
from mlem.core.errors import DeploymentError, EndpointNotFound, MlemError
from mlem.core.objects import (
    DeployState,
    DeployStatus,
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


class K8sEnv(MlemEnv):
    """MlemEnv implementation for Kubernetes Environments"""

    type: ClassVar = "kubernetes"
    """Type of deployment being used for the Kubernetes environment"""

    registry: Optional[DockerRegistry] = None
    """Docker registry"""
    templates_dir: List[str] = []
    """List of dirs where templates reside"""


class K8sDeploymentState(DeployState):
    """DeployState implementation for Kubernetes deployments"""

    type: ClassVar = "kubernetes"

    image: Optional[DockerImage] = None
    """Docker Image being used for Deployment"""
    deployment_name: Optional[str] = None
    """Name of Deployment"""


class K8sDeployment(
    MlemDeployment[K8sDeploymentState, K8sEnv], K8sYamlBuildArgs
):
    """MlemDeployment implementation for Kubernetes deployments"""

    type: ClassVar = "kubernetes"
    state_type: ClassVar = K8sDeploymentState
    """Type of state for Kubernetes deployments"""
    env_type: ClassVar = K8sEnv

    server: Optional[Server] = None
    """Type of Server to use, with options such as FastAPI, RabbitMQ etc."""
    registry: Optional[DockerRegistry] = DockerRegistry()
    """Docker registry"""
    daemon: Optional[DockerDaemon] = DockerDaemon(host="")
    """Docker daemon"""
    kube_config_file_path: Optional[str] = None
    """Path for kube config file of the cluster"""
    templates_dir: List[str] = []
    """List of dirs where templates reside"""

    def load_kube_config(self):
        config.load_kube_config(
            config_file=self.kube_config_file_path
            or os.getenv("KUBECONFIG", default="~/.kube/config")
        )

    def _get_client(self, state: K8sDeploymentState) -> Client:
        self.load_kube_config()
        service = client.CoreV1Api().list_namespaced_service(self.namespace)
        try:
            host, port = self.service_type.get_host_and_port(
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

    def get_registry(self):
        registry = self.registry or self.get_env().registry
        if not registry:
            raise MlemError(
                "registry to be used by Docker is not set or supplied"
            )
        return registry

    def get_image_name(self):
        return self.image_name or generate_docker_container_name()

    def get_server(self):
        return (
            self.server
            or project_config(
                self.loc.project if self.is_saved else None
            ).server
        )

    def deploy(self, model: MlemModel):
        redeploy = False
        with self.lock_state():
            self.load_kube_config()
            state: K8sDeploymentState = self.get_state()
            if state.image is None or self.model_changed(model):
                image_name = self.get_image_name()
                state.image = build_k8s_docker(
                    meta=model,
                    image_name=image_name,
                    registry=self.get_registry(),
                    daemon=self.daemon,
                    server=self.get_server(),
                )
                state.update_model(model)
                redeploy = True

            if (
                state.deployment_name is None or redeploy
            ) and state.image is not None:
                generator = K8sYamlGenerator(
                    namespace=self.namespace,
                    image_name=state.image.name,
                    image_uri=state.image.uri,
                    image_pull_policy=self.image_pull_policy,
                    port=self.port,
                    service_type=self.service_type,
                    templates_dir=self.templates_dir
                    or self.get_env().templates_dir,
                )
                create_k8s_resources(generator)

                if pod_is_running(namespace=self.namespace):
                    deployments_list = (
                        client.AppsV1Api().list_namespaced_deployment(
                            namespace=self.namespace
                        )
                    )

                    if len(deployments_list.items) == 0:
                        raise DeploymentError(
                            f"Deployment {image_name} couldn't be found in {self.namespace} namespace"
                        )
                    dpl_name = deployments_list.items[0].metadata.name
                    state.deployment_name = dpl_name
                    self.update_state(state)

                    echo(
                        EMOJI_OK
                        + f"Deployment {state.deployment_name} is up in {self.namespace} namespace"
                    )
                else:
                    raise DeploymentError(
                        f"Deployment {image_name} couldn't be set-up on the Kubernetes cluster"
                    )

    def remove(self):
        with self.lock_state():
            self.load_kube_config()
            state: K8sDeploymentState = self.get_state()
            if state.deployment_name is not None:
                client.CoreV1Api().delete_namespace(name=self.namespace)
                if namespace_deleted(self.namespace):
                    echo(
                        EMOJI_OK
                        + f"Deployment {state.deployment_name} and the corresponding service are removed from {self.namespace} namespace"
                    )
                    state.deployment_name = None
                    self.update_state(state)

    def get_status(self, raise_on_error=True) -> DeployStatus:
        self.load_kube_config()
        state: K8sDeploymentState = self.get_state()
        if state.deployment_name is None:
            return DeployStatus.NOT_DEPLOYED

        pods_list = client.CoreV1Api().list_namespaced_pod(self.namespace)

        return POD_STATE_MAPPING[pods_list.items[0].status.phase]
