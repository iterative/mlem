import contextlib
import json
import logging
import os
import tempfile
import time
from collections import defaultdict
from time import sleep
from typing import (
    ClassVar,
    Dict,
    Generator,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)

import docker
import requests
from docker import errors
from docker.errors import NotFound
from pydantic import BaseModel

from mlem.config import LOCAL_CONFIG, project_config
from mlem.contrib.docker.context import DockerBuildArgs, DockerModelDirectory
from mlem.contrib.docker.utils import (
    build_image_with_logs,
    container_is_running,
    container_logs,
    container_status,
    create_docker_client,
    image_exists_at_dockerhub,
    print_docker_logs,
    wrap_docker_error,
)
from mlem.core.base import MlemABC
from mlem.core.errors import DeploymentError
from mlem.core.objects import (
    DeployState,
    DeployStatus,
    MlemBuilder,
    MlemDeployment,
    MlemEnv,
    MlemModel,
)
from mlem.runtime.server import Server
from mlem.ui import EMOJI_BUILD, EMOJI_OK, EMOJI_UPLOAD, echo, set_offset

logger = logging.getLogger(__name__)


CONTAINER_STATUS_MAPPING = {
    "created": DeployStatus.NOT_DEPLOYED,
    "running": DeployStatus.RUNNING,
    "restarting": DeployStatus.STARTING,
    "paused": DeployStatus.STOPPED,
    "exited": DeployStatus.CRASHED,
    "dead": DeployStatus.CRASHED,
}


class DockerRegistry(MlemABC):
    """Registry for docker images. This is the default implementation that represents registry of the docker daemon"""

    abs_name: ClassVar = "docker_registry"
    type: ClassVar = "local"

    class Config:
        type_root = True

    def get_host(self) -> Optional[str]:
        """Returns registry host or emty string for local"""
        return ""

    def push(self, client: docker.DockerClient, tag: str):
        """Pushes image to registry

        :param client: DockerClient to use
        :param tag: name of the tag to push"""

    def login(self, client: docker.DockerClient):
        """Login to registry

        :param client: DockerClient to use"""

    def uri(self, image: str):
        """Cretate an uri for image in this registry

        :param image: image name"""
        return image

    def image_exists(self, client: docker.DockerClient, image: "DockerImage"):
        """Check if image exists in this registry

        :param client: DockerClient to use
        :param image: :class:`.DockerImage` to check"""
        try:
            client.images.get(image.uri)
            return True
        except errors.ImageNotFound:
            return False

    def delete_image(
        self,
        client: docker.DockerClient,
        image: "DockerImage",
        force: bool = False,
        **kwargs,
    ):
        """Deleta image from this registry

        :param client: DockerClient to use
        :param image: :class:`.DockerImage` to delete
        :param force: force delete
        """
        try:
            client.images.remove(image.uri, force=force, **kwargs)
        except errors.ImageNotFound:
            pass


class DockerIORegistry(DockerRegistry):
    """The class represents docker.io registry."""

    type: ClassVar = "docker_io"

    def get_host(self) -> Optional[str]:
        return "https://index.docker.io/v1/"

    def push(self, client, tag):
        client.images.push(tag)
        echo(EMOJI_UPLOAD + f"Pushed image {tag} to docker.io")

    def image_exists(self, client, image: "DockerImage"):
        return image_exists_at_dockerhub(image.uri)

    def delete_image(
        self, client, image: "DockerImage", force=False, **kwargs
    ):
        logger.warning("Skipping deleting image %s from docker.io", image.name)


class RemoteRegistry(DockerRegistry):
    """DockerRegistry implementation for official Docker Registry (as in https://docs.docker.com/registry/)"""

    type: ClassVar = "remote"
    # TODO: https://github.com/iterative/mlem/issues/38 credentials
    host: Optional[str] = None
    """Address of the registry"""

    def login(self, client):
        """
        Logs in to Docker registry

        Corresponding credentials should be specified as environment variables per registry:
        e.g., if registry host is "168.32.25.1:5000" then
        "168_32_25_1_5000_USERNAME" and "168_32_25_1_5000_PASSWORD" variables should be specified

        :param client: Docker client instance
        :return: nothing
        """

        host_for_env = self.host.replace(".", "_").replace(":", "_")
        username_var = f"{host_for_env}_username".upper()
        username = os.getenv(username_var)
        password_var = f"{host_for_env}_password".upper()
        password = os.getenv(password_var)

        if username and password:
            self._login(self.host, client, username, password)
            logger.debug("Logged in to remote registry at host %s", self.host)
        else:
            logger.warning(
                "Skipped logging in to remote registry at host %s because no credentials given. "
                "You could specify credentials as %s and %s environment variables.",
                self.host,
                username_var,
                password_var,
            )

    @staticmethod
    def _login(host, client: docker.DockerClient, username, password):
        res = client.login(
            registry=host, username=username, password=password, reauth=True
        )
        if res["Status"] != "Login Succeeded":
            raise DeploymentError(f"Cannot login to registry: {res}")

    def get_host(self) -> Optional[str]:
        return self.host

    def push(self, client, tag):
        echo(EMOJI_UPLOAD + f"Pushing image {tag} to {self.get_host()}")
        res = client.images.push(tag)
        for line in res.splitlines():
            status = json.loads(line)
            if "error" in status:
                error_msg = status["error"]
                raise DeploymentError(f"Cannot push docker image: {error_msg}")
        echo(EMOJI_OK + f"Pushed image {tag} to {self.get_host()}")

    def uri(self, image: str):
        return f"{self.get_host()}/{image}"

    def _get_digest(self, name, tag):
        r = requests.head(
            f"http://{self.host}/v2/{name}/manifests/{tag}",
            headers={
                "Accept": "application/vnd.docker.distribution.manifest.v2+json"
            },
        )
        if r.status_code != 200:
            return None
        return r.headers["Docker-Content-Digest"]

    def image_exists(self, client, image: "DockerImage"):
        name = image.fullname
        digest = self._get_digest(name, image.tag)
        if digest is None:
            return False
        r = requests.head(f"http://{self.host}/v2/{name}/manifests/{digest}")
        if r.status_code == 404:
            return False
        if r.status_code == 200:
            return True
        r.raise_for_status()
        raise ValueError(
            "Response did not return code 200, but not raised an exception"
        )

    def delete_image(
        self, client, image: "DockerImage", force=False, **kwargs
    ):
        name = image.fullname
        digest = self._get_digest(name, image.tag)
        if digest is None:
            return
        requests.delete(f"http://{self.host}/v2/{name}/manifests/{digest}")


class DockerDaemon(MlemABC):
    """Class that represents docker daemon"""

    host: str = (
        ""  # TODO: https://github.com/iterative/mlem/issues/38 credentials
    )
    """adress of the docker daemon (empty string for local)"""

    @contextlib.contextmanager
    def client(self) -> Iterator[docker.DockerClient]:
        """Get DockerClient isntance"""
        with create_docker_client(self.host) as c:
            yield c


class DockerImage(BaseModel):
    """:class:`.Image.Params` implementation for docker images
    full uri for image looks like registry.host/repository/name:tag"""

    name: str
    """name of the image"""
    tag: str = "latest"
    """tag of the image"""
    repository: Optional[str] = None
    """repository of the image"""
    registry: DockerRegistry = DockerRegistry()
    """DockerRegistry instance with this image"""
    image_id: Optional[str] = None
    """internal docker id of this image"""

    @property
    def fullname(self):
        return (
            f"{self.repository}/{self.name}"
            if self.repository is not None
            else self.name
        )

    @property
    def uri(self) -> str:
        return self.registry.uri(f"{self.fullname}:{self.tag}")

    def exists(self, client: docker.DockerClient):
        """Checks if this image exists in it's registry"""
        return self.registry.image_exists(client, self)

    def delete(self, client: docker.DockerClient, force=False, **kwargs):
        """Deletes image from registry"""
        self.registry.delete_image(client, self, force, **kwargs)


class DockerEnv(MlemEnv):
    """MlemEnv implementation for docker environment"""

    type: ClassVar = "docker"
    registry: DockerRegistry = DockerRegistry()
    """Default registry to push images to"""
    daemon: DockerDaemon = DockerDaemon(host="")
    """Docker daemon parameters"""

    def delete_image(self, image: DockerImage, force: bool = False, **kwargs):
        with self.daemon.client() as client:
            return image.delete(client, force=force, **kwargs)

    def image_exists(self, image: DockerImage):
        with self.daemon.client() as client:
            return image.exists(client)


class DockerContainerState(DeployState):
    """State of docker container deployment"""

    type: ClassVar = "docker_container"

    image: Optional[DockerImage]
    """Built image"""
    container_name: Optional[str]
    """Name of container"""
    container_id: Optional[str]
    """Started container id"""


class _DockerBuildMixin(BaseModel):
    server: Optional[Server] = None
    """Server to use"""
    args: DockerBuildArgs = DockerBuildArgs()
    """Additional docker arguments"""


def generate_docker_container_name():
    return f"mlem-deploy-{int(time.time())}"


class DockerContainer(
    MlemDeployment[DockerContainerState, DockerEnv], _DockerBuildMixin
):
    """MlemDeployment implementation for docker containers"""

    type: ClassVar = "docker_container"
    state_type: ClassVar = DockerContainerState
    env_type: ClassVar = DockerEnv

    container_name: Optional[str] = None
    """Name to use for container"""
    image_name: Optional[str] = None
    """Name to use for image"""
    ports: List[str] = []
    """Publish container ports. See https://docs.docker.com/config/containers/container-networking/#published-ports"""
    params: Dict[str, str] = {}
    """Additional params"""
    rm: bool = True
    """Remove container on stop"""

    def get_port_mapping(
        self,
    ) -> Dict[
        Union[int, str], Union[int, str, None, Tuple[str, int], List[int]]
    ]:
        container_ports: Dict[
            Union[int, str], List[Union[int, str, None, Tuple[str, int]]]
        ] = defaultdict(list)
        cport: Union[int, str]
        for str_port in self.ports:
            parts = str_port.split(":")
            if len(parts) == 1:
                # 80 -> container:80 - host:random -> {80: None}
                cport = parts[0]
                cport = int(cport) if cport.isnumeric() else cport
                container_ports[cport].append(None)
            elif len(parts) == 2:
                # 8080:80 -> container:80 - host:8080 -> {80: 8080}
                # 8080:80/udp -> container:80/udp - host:8080 -> {'80/udp': 8080}
                hport, cport = parts
                cport = int(cport) if cport.isnumeric() else cport
                container_ports[cport].append(int(hport))
            elif len(parts) == 3:
                # 192.168.1.100:8080:80 -> container:80 - 192.168.1.100:8080 -> {80: ('192.168.1.100', 8080)}
                iface, hport, cport = parts
                cport = int(cport) if cport.isnumeric() else cport
                container_ports[cport].append((iface, int(hport)))
            else:
                raise ValueError(
                    f"Unable to parse '{str_port}' for port mapping"
                )
        return {
            cport: ports if len(ports) > 1 else ports[0]
            for cport, ports in container_ports.items()
        }

    @property
    def ensure_image_name(self):
        return self.image_name or self.container_name

    def _get_client(self, state: DockerContainerState):
        raise NotImplementedError

    def deploy(self, model: MlemModel):
        # self.check_type(meta)
        redeploy = False
        with self.lock_state():
            state = self.get_state()
            if state.image is None or self.model_changed(model):
                from .helpers import build_model_image

                image_name = (
                    self.image_name
                    or self.container_name
                    or generate_docker_container_name()
                )
                echo(EMOJI_BUILD + f"Creating docker image {image_name}")
                with set_offset(2):
                    state.image = build_model_image(
                        model,
                        image_name,
                        self.server
                        or project_config(
                            self.loc.project if self.is_saved else None
                        ).server,
                        self.get_env(),
                        force_overwrite=True,
                        **self.args.dict(),
                    )
                state.update_model(model)
                self.update_state(state)
                redeploy = True
            if state.container_id is None or redeploy:
                self.run_container(state)

            echo(EMOJI_OK + f"Container {state.container_name} is up")

    def remove(self):
        with self.lock_state():
            state = self.get_state()
            if state.container_id is None:
                raise DeploymentError(
                    f"Container {self.container_name} is not deployed"
                )

            with self.get_env().daemon.client() as client:
                try:
                    container = client.containers.get(state.container_id)
                    container.stop()
                    container.remove()
                except docker.errors.NotFound:
                    pass
            state.container_id = None
            self.update_state(state)

    def get_status(self, raise_on_error=True) -> DeployStatus:
        state = self.get_state()
        if state.container_id is None:
            return DeployStatus.NOT_DEPLOYED

        with self.get_env().daemon.client() as client:
            try:
                status = container_status(client, state.container_id)
                return CONTAINER_STATUS_MAPPING[status]
            except NotFound:
                return DeployStatus.UNKNOWN

    def run_container(
        self,
        state: Optional[DockerContainerState] = None,
    ):
        state = state or self.get_state()
        if state.image is None:
            raise DeploymentError(
                f"Image {self.ensure_image_name} is not built"
            )

        with self.get_env().daemon.client() as client:
            state.image.registry.login(client)

            try:
                # always detach from container and just stream logs if detach=False
                name = self.container_name or generate_docker_container_name()
                container = client.containers.run(
                    state.image.uri,
                    name=name,
                    auto_remove=self.rm,
                    ports=self.get_port_mapping(),
                    detach=True,
                    **self.params,
                )
                state.container_id = container.id
                state.container_name = name
                self.update_state(state)
                sleep(0.5)
                if not container_is_running(client, name):
                    if not self.rm:
                        for log in self.logs(stdout=False, stderr=True):
                            raise DeploymentError(
                                "The container died unexpectedly.", log
                            )
                    else:
                        # Can't get logs from removed container
                        raise DeploymentError(
                            "The container died unexpectedly. Try to run the container "
                            "with rm=False args to get more info."
                        )
            except docker.errors.ContainerError as e:
                raise DeploymentError(
                    "Docker container raised an error: " + e.stderr.decode()
                ) from e

    def logs(self, **kwargs) -> Generator[str, None, None]:
        state = self.get_state()
        if state.container_id is None:
            raise DeploymentError(
                f"Container {self.container_name} is not deployed"
            )
        with self.get_env().daemon.client() as client:
            container = client.containers.get(state.container_id)
            yield from container_logs(container, **kwargs)


class DockerDirBuilder(MlemBuilder, _DockerBuildMixin):
    """Create a directory with docker context to build docker image"""

    type: ClassVar[str] = "docker_dir"
    target: str
    """Path to save result"""

    def build(self, obj: MlemModel):
        docker_dir = DockerModelDirectory(
            model=obj,
            server=self.server or LOCAL_CONFIG.server,
            path=self.target,
            docker_args=self.args,
            debug=True,
        )
        docker_dir.write_distribution()
        return docker_dir


class DockerImageBuilder(MlemBuilder, _DockerBuildMixin):
    """Build docker image from model"""

    type: ClassVar[str] = "docker"
    image: DockerImage
    """Image parameters"""
    env: DockerEnv = DockerEnv()
    """Where to build and push image. Defaults to local docker daemon"""
    force_overwrite: bool = False
    """Ignore existing image with same name"""
    push: bool = True
    """Push image to registry after it is built"""

    def build(self, obj: MlemModel) -> DockerImage:
        with tempfile.TemporaryDirectory(prefix="mlem_build_") as tempdir:
            if self.args.prebuild_hook is not None:
                self.args.prebuild_hook(  # pylint: disable=not-callable # but it is
                    self.args.python_version
                )
            DockerDirBuilder(
                server=self.server, args=self.args, target=tempdir
            ).build(obj)

            return self.build_image(tempdir)

    @wrap_docker_error
    def build_image(self, context_dir: str) -> DockerImage:
        tag = self.image.uri
        logger.debug("Building docker image %s from %s...", tag, context_dir)
        echo(EMOJI_BUILD + f"Building docker image {tag}...")
        with self.env.daemon.client() as client:
            if self.push:
                self.image.registry.login(client)

            if self.force_overwrite:
                self.image.delete(client)  # to avoid spawning dangling images
            elif self.image.exists(client):
                raise ValueError(
                    f"Image {tag} already exists at {self.image.registry}. "
                    f"Change name or set force_overwrite=True."
                )

            try:
                image, _ = build_image_with_logs(
                    client,
                    path=context_dir,
                    tag=tag,
                    rm=True,
                    platform=self.args.platform,
                )
                self.image.image_id = image.id
                echo(EMOJI_OK + f"Built docker image {tag}")

                if self.push:
                    self.image.registry.push(client, tag)

                return self.image
            except errors.BuildError as e:
                print_docker_logs(e.build_log, logging.ERROR)
                raise


# Copyright 2019 Zyfra
# Copyright 2021 Iterative
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
