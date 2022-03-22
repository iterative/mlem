import contextlib
import json
import logging
import os
import tempfile
from typing import ClassVar, Dict, Iterator, Optional

import docker
import requests
from docker import errors
from pydantic import BaseModel

from mlem.contrib.docker.context import DockerBuildArgs, DockerModelDirectory
from mlem.contrib.docker.utils import (
    build_image_with_logs,
    create_docker_client,
    image_exists_at_dockerhub,
    print_docker_logs,
)
from mlem.core.base import MlemObject
from mlem.core.errors import DeploymentError
from mlem.core.objects import ModelMeta
from mlem.pack import Packager
from mlem.runtime.server.base import Server

logger = logging.getLogger(__name__)


class DockerRegistry(MlemObject):
    """Registry for docker images. This is the default implementation that represents registry of the docker daemon"""

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

    def get_host(self) -> Optional[str]:
        return "https://index.docker.io/v1/"

    def push(self, client, tag):
        client.images.push(tag)
        logger.info("Pushed image %s to docker.io", tag)

    def image_exists(self, client, image: "DockerImage"):
        return image_exists_at_dockerhub(image.uri)

    def delete_image(
        self, client, image: "DockerImage", force=False, **kwargs
    ):
        logger.warning("Skipping deleting image %s from docker.io", image.name)


class RemoteRegistry(DockerRegistry):
    """DockerRegistry implementation for official Docker Registry (as in https://docs.docker.com/registry/)

    :param host: adress of the registry"""

    host: Optional[
        str
    ] = None  # TODO: https://github.com/iterative/mlem/issues/38 credentials

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
            logger.info("Logged in to remote registry at host %s", self.host)
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
        res = client.images.push(tag)
        for line in res.splitlines():
            status = json.loads(line)
            if "error" in status:
                error_msg = status["error"]
                auth = (
                    client.api._auth_configs  # pylint: disable=protected-access
                )
                raise DeploymentError(
                    f"Cannot push docker image: {error_msg} {auth}"
                )
        logger.info(
            "Pushed image %s to remote registry at host %s", tag, self.host
        )

    def uri(self, image: str):
        return f"{self.host}/{image}"

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


class DockerDaemon(MlemObject):
    """Class that represents docker daemon

    :param host: adress of the docker daemon (empty string for local)"""

    host: str  # TODO: https://github.com/iterative/mlem/issues/38 credentials

    @contextlib.contextmanager
    def client(self) -> Iterator[docker.DockerClient]:
        """Get DockerClient isntance"""
        with create_docker_client(self.host) as c:
            yield c


class DockerImage(BaseModel):
    """:class:`.Image.Params` implementation for docker images
    full uri for image looks like registry.host/repository/name:tag

    :param name: name of the image
    :param tag: tag of the image
    :param repository: repository of the image
    :param registry: :class:`.DockerRegistry` instance with this image
    :param image_id: docker internal id of this image"""

    name: str
    tag: str = "latest"
    repository: Optional[str] = None
    registry: DockerRegistry = DockerRegistry()
    image_id: Optional[str] = None

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


class DockerContainer(BaseModel):
    """:class:`.RuntimeInstance.Params` implementation for docker containers

    :param name: name of the container
    :param port_mapping: port mapping in this container
    :param params: other parameters for docker run cmd
    :param container_id: internal docker id for this container"""

    name: str
    port_mapping: Dict[int, int] = {}
    params: Dict[str, str] = {}
    container_id: Optional[str] = None


class DockerEnv(BaseModel):
    """:class:`.RuntimeEnvironment.Params` implementation for docker environment

    :param registry: default registry to push images to
    :param daemon: :class:`.DockerDaemon` instance"""

    registry: DockerRegistry = DockerRegistry()
    daemon: DockerDaemon = DockerDaemon(host="")

    def delete_image(self, image: DockerImage, force: bool = False, **kwargs):
        with self.daemon.client() as client:
            return image.delete(client, force=force, **kwargs)

    def image_exists(self, image: DockerImage):
        with self.daemon.client() as client:
            return image.exists(client)


class DockerDirPackager(Packager):
    type: ClassVar[str] = "docker_dir"
    server: Server
    args: DockerBuildArgs = DockerBuildArgs()

    def package(self, obj: ModelMeta, out: str):
        docker_dir = DockerModelDirectory(
            model=obj,
            server=self.server,
            path=out,
            docker_args=self.args,
            debug=True,
        )
        docker_dir.write_distribution()
        return docker_dir


class DockerImagePackager(DockerDirPackager):
    type: ClassVar[str] = "docker"
    image: DockerImage
    env: DockerEnv = DockerEnv()
    force_overwrite: bool = False
    push: bool = True

    def package(self, obj: ModelMeta, out: str) -> DockerImage:
        with tempfile.TemporaryDirectory(prefix="mlem_build_") as tempdir:
            if self.args.prebuild_hook is not None:
                self.args.prebuild_hook(  # pylint: disable=not-callable # but it is
                    self.args.python_version
                )
            super().package(obj, tempdir)
            if not self.image.name:
                # TODO: https://github.com/iterative/mlem/issues/65
                self.image.name = out

            return self.build(tempdir)

    def build(self, context_dir: str) -> DockerImage:
        tag = self.image.uri
        logger.debug("Building docker image %s from %s...", tag, context_dir)
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
                logger.info("Built docker image %s", tag)

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
