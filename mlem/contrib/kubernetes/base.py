import os
import tempfile
from time import sleep
from typing import ClassVar, Optional

from kubernetes import client, config, utils, watch

from mlem.config import project_config
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
from mlem.ui import EMOJI_BUILD, EMOJI_OK, echo, set_offset

from ..docker.base import (
    DockerDaemon,
    DockerEnv,
    DockerImage,
    DockerRegistry,
    generate_docker_container_name,
)
from ..docker.helpers import build_model_image
from .context import K8sYamlBuildArgs, K8sYamlGenerator

POD_STATE_MAPPING = {
    "Pending": DeployStatus.STARTING,
    "Running": DeployStatus.RUNNING,
    "Succeeded": DeployStatus.STOPPED,
    "Failed": DeployStatus.CRASHED,
    "Unknown": DeployStatus.UNKNOWN,
}


class K8sDeploymentState(DeployState):
    type: ClassVar = "kubernetes"
    image: Optional[DockerImage] = None
    deployment_name: Optional[str] = None


class K8sDeployment(MlemDeployment, K8sYamlBuildArgs):
    type: ClassVar = "kubernetes"
    server: Optional[Server] = None
    state_type: ClassVar = K8sDeploymentState
    kube_config_file_path: Optional[str] = None

    def load_kube_config(self):
        config.load_kube_config(
            config_file=self.kube_config_file_path
            or os.getenv("KUBECONFIG", default="~/.kube/config")
        )

    def _get_client(self, state: K8sDeploymentState) -> Client:
        self.load_kube_config()
        service = client.CoreV1Api().list_namespaced_service(
            f"mlem-{state.deployment_name}-app"
        )
        port = service.items[0].spec.ports[0].port
        host = service.items[0].status.load_balancer.ingress[0].ip
        return HTTPClient(host=host, port=port)


class K8sEnv(MlemEnv[K8sDeployment]):
    type: ClassVar = "kubernetes"
    deploy_type: ClassVar = K8sDeployment
    registry: Optional[DockerRegistry] = DockerRegistry()
    daemon: Optional[DockerDaemon] = DockerDaemon(
        host=os.getenv("DOCKER_HOST", default="")
    )

    def create_k8s_resources(self, generator):
        k8s_client = client.ApiClient()
        with tempfile.TemporaryDirectory(
            prefix="mlem_k8s_yaml_build_"
        ) as tempdir:
            filename = os.path.join(tempdir, "resource.yaml")
            generator.write(filename)
            utils.create_from_yaml(k8s_client, filename, verbose=True)

    def deploy(self, meta: K8sDeployment):
        self.check_type(meta)
        redeploy = False
        with meta.lock_state():
            meta.load_kube_config()
            state: K8sDeploymentState = meta.get_state()
            if state.image is None or meta.model_changed():

                image_name = (
                    meta.image_name or generate_docker_container_name()
                )
                echo(EMOJI_BUILD + f"Creating docker image {image_name}")
                with set_offset(2):
                    state.image = build_model_image(
                        meta.get_model(),
                        image_name,
                        meta.server
                        or project_config(
                            meta.loc.project if meta.is_saved else None
                        ).server,
                        DockerEnv(registry=self.registry, daemon=self.daemon),
                        force_overwrite=True,
                    )
                meta.update_model_hash(state=state)
                redeploy = True

            if state.deployment_name is None or redeploy:
                k8s_yaml_builder = K8sYamlBuilder(
                    image=state.image,
                    image_pull_policy=meta.image_pull_policy,
                    port=meta.port,
                    service_type=meta.service_type,
                )
                generator = K8sYamlGenerator(
                    **k8s_yaml_builder.get_resource_args().dict()
                )
                self.create_k8s_resources(generator)

                w = watch.Watch()
                for event in w.stream(
                    func=client.CoreV1Api().list_namespaced_pod,
                    namespace=f"mlem-{meta.image_name}-app",
                    timeout_seconds=60,
                ):
                    if event["object"].status.phase == "Running":
                        w.stop()

                deployments_list = (
                    client.AppsV1Api().list_namespaced_deployment(
                        namespace=f"mlem-{meta.image_name}-app"
                    )
                )

                state.deployment_name = deployments_list.items[0].metadata.name
                meta.update_state(state)

            echo(
                EMOJI_OK
                + f"Deployment {state.deployment_name} is up in mlem-{meta.image_name}-app namespace"
            )

    def remove(self, meta: K8sDeployment):
        self.check_type(meta)
        with meta.lock_state():
            meta.load_kube_config()
            state: K8sDeploymentState = meta.get_state()
            if state.deployment_name is not None:
                client.AppsV1Api().delete_namespaced_deployment(
                    name=state.deployment_name,
                    namespace=f"mlem-{meta.image_name}-app",
                )
                client.CoreV1Api().delete_namespaced_service(
                    name=state.deployment_name,
                    namespace=f"mlem-{meta.image_name}-app",
                )
                client.CoreV1Api().delete_namespace(
                    name=f"mlem-{meta.image_name}-app"
                )
                sleep(0.5)
            echo(
                EMOJI_OK
                + f"Deployment {state.deployment_name} and the corresponding service are removed from mlem-{meta.image_name}-app namespace"
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

        pods_list = client.CoreV1Api().list_namespaced_pod(
            f"mlem-{meta.image_name}-app"
        )

        return POD_STATE_MAPPING[pods_list.items[0].status.phase]


class K8sYamlBuilder(MlemBuilder, K8sYamlBuildArgs):
    type: ClassVar = "kubernetes"
    image: DockerImage

    def get_resource_args(self):
        return K8sYamlBuildArgs(
            image_name=self.image.name,
            image_uri=self.image.uri,
            image_pull_policy=self.image_pull_policy,
            port=self.port,
            service_type=self.service_type,
        )

    def build(self, obj: MlemModel):
        resource_args = self.get_resource_args()

        generator = K8sYamlGenerator(**resource_args.dict())
        resource_yaml = generator.generate()

        generator.write("resources.yaml")
        echo(
            EMOJI_OK
            + f"resources.yaml generated for {obj.basename}, apply manualy using kubectl OR use mlem deploy"
        )

        return resource_yaml
