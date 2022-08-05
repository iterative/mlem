import os
from typing import ClassVar, Optional

from kubernetes import config

from mlem.config import project_config
from mlem.core.objects import (
    DeployState,
    DeployStatus,
    MlemDeployment,
    MlemEnv,
    MlemModel,
)
from mlem.runtime.server import Server
from mlem.ui import EMOJI_BUILD, EMOJI_OK, echo, set_offset

from ..docker.base import (
    DockerEnv,
    DockerImage,
    DockerImageBuilder,
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
    pod_name: Optional[str]
    kube_config_file_path: Optional[str] = None

    def get_client(self):
        config.load_kube_config(
            config_file=self.kube_config_file_path
            or os.getenv("KUBECONFIG", default="~/.kube/config")
        )


class K8sDeployment(MlemDeployment):
    type: ClassVar = "kubernetes"
    server: Optional[Server] = None
    state_type: ClassVar = K8sDeploymentState
    image_name: Optional[str] = None


class K8sEnv(MlemEnv[K8sDeployment]):
    type: ClassVar = "kubernetes"
    deploy_type: ClassVar = K8sDeployment
    registry: Optional[DockerRegistry] = DockerRegistry()

    def create_k8s_resources():
        pass

    def deploy(self, meta: K8sDeployment):
        self.check_type(meta)
        redeploy = False
        with meta.lock_state():
            state: K8sDeploymentState = meta.get_state()
            if state.image is None or meta.model_changed():
                from ..docker.helpers import build_model_image

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
                        DockerEnv(registry=self.registry),
                        force_overwrite=True,
                    )
                meta.update_model_hash(state=state)
                redeploy = True
            if state.pod_name is None or redeploy:
                state.pod_name = "..."  # TODO: use K8 APIs to deploy the yaml
                meta.update_state(state)

            echo(EMOJI_OK + f"Pod {state.pod_name} is up")

    def remove(self, meta: K8sDeployment):
        self.check_type(meta)
        with meta.lock_state():
            state: K8sDeploymentState = meta.get_state()
            if state.pod_name is not None:
                # use kubernetes client to delete deployment and service
                ...
            state.pod_name = None
            meta.update_state(state)

    def get_status(
        self, meta: K8sDeployment, raise_on_error=True
    ) -> DeployStatus:
        self.check_type(meta)
        state: K8sDeploymentState = meta.get_state()
        if state.pod_name is None:
            return DeployStatus.NOT_DEPLOYED

        return DeployStatus.STOPPED  # replace it by actual value
        # get state from kubernetes client and use POD_STATE_MAPPING to return


class K8sYamlBuilder(DockerImageBuilder):
    type: ClassVar = "kubernetes"

    def build(self, obj: MlemModel):
        image = super().build(obj)
        resource_args = K8sYamlBuildArgs(
            image=image.uri,
        )
        resource_yaml = K8sYamlGenerator(**resource_args.dict()).generate()

        return resource_yaml
