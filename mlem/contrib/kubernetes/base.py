import os
import tempfile
from time import sleep
from typing import ClassVar, Optional

from kubernetes import client, config, utils

from mlem.config import project_config
from mlem.core.objects import (
    DeployState,
    DeployStatus,
    MlemBuilder,
    MlemDeployment,
    MlemEnv,
    MlemModel,
)
from mlem.runtime.server import Server
from mlem.ui import EMOJI_BUILD, EMOJI_OK, echo, set_offset

from ..docker.base import (
    DockerEnv,
    DockerImage,
    DockerRegistry,
    generate_docker_container_name,
)
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
    image: Optional[DockerImage]
    deployment_name: Optional[str]


class K8sDeployment(MlemDeployment, K8sYamlBuildArgs):
    type: ClassVar = "kubernetes"
    server: Optional[Server] = None
    state_type: ClassVar = K8sDeploymentState
    kube_config_file_path: Optional[str] = None

    def _get_client(self, state: K8sDeploymentState):
        config.load_kube_config(
            config_file=self.kube_config_file_path
            or os.getenv("KUBECONFIG", default="~/.kube/config")
        )


class K8sEnv(MlemEnv[K8sDeployment]):
    type: ClassVar = "kubernetes"
    deploy_type: ClassVar = K8sDeployment
    registry: Optional[DockerRegistry] = DockerRegistry()

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
            meta.get_client()
            state: K8sDeploymentState = meta.get_state()
            if state.image is None or meta.model_changed():
                from ..docker.helpers import build_model_image

                image_name = meta.name or generate_docker_container_name()
                echo(EMOJI_BUILD + f"Creating docker image {image_name}")
                with set_offset(2):
                    state.image = build_model_image(
                        meta.get_model(),
                        image_name,
                        meta.server
                        or project_config(
                            meta.loc.project if meta.is_saved else None
                        ).server,
                        DockerEnv(registry=self.registry),
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

                sleep(0.5)

                deployments_list = (
                    client.AppsV1Api().list_namespaced_deployment(
                        namespace=f"mlem-{meta.name}-app"
                    )
                )

                state.deployment_name = deployments_list.items[0].metadata.name
                meta.update_state(state)

            echo(
                EMOJI_OK
                + f"Deployment {state.deployment_name} is up in mlem-{meta.name}-app namespace"
            )

    def remove(self, meta: K8sDeployment):
        self.check_type(meta)
        with meta.lock_state():
            meta.get_client()
            state: K8sDeploymentState = meta.get_state()
            if state.deployment_name is not None:
                client.AppsV1Api().delete_namespaced_deployment(
                    name=state.deployment_name,
                    namespace=f"mlem-{meta.name}-app",
                )
                client.CoreV1Api().delete_namespaced_service(
                    name=state.deployment_name,
                    namespace=f"mlem-{meta.name}-app",
                )
                client.CoreV1Api().delete_namespace(
                    name=f"mlem-{meta.name}-app"
                )
            echo(
                EMOJI_OK
                + f"Deployment {state.deployment_name} and the corresponding service are removed from mlem-{meta.name}-app namespace"
            )
            state.deployment_name = None
            meta.update_state(state)

    def get_status(
        self, meta: K8sDeployment, raise_on_error=True
    ) -> DeployStatus:
        self.check_type(meta)
        meta.get_client()
        state: K8sDeploymentState = meta.get_state()
        if state.deployment_name is None:
            return DeployStatus.NOT_DEPLOYED

        pods_list = client.CoreV1Api().list_namespaced_pod(
            f"mlem-{meta.name}-app"
        )

        return POD_STATE_MAPPING[pods_list.items[0].status.phase]


class K8sYamlBuilder(MlemBuilder, K8sYamlBuildArgs):
    type: ClassVar = "kubernetes"
    image: DockerImage

    def get_resource_args(self):
        return K8sYamlBuildArgs(
            name=self.image.name,
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
