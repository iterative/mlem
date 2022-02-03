import itertools
import logging
import os
import re
import time
from contextlib import contextmanager
from threading import Lock
from typing import Any, Iterator, Tuple, Union

import docker
import requests
from docker.errors import BuildError, DockerException
from docker.models.images import Image
from docker.utils.json_stream import json_stream

logger = logging.getLogger(__name__)


def print_docker_logs(logs, level=logging.DEBUG):
    for log in logs:
        if "stream" in log:
            logger.log(level, str(log["stream"]).strip())
        else:
            logger.log(level, str(log).strip())


def build_image_with_logs(
    client: docker.DockerClient,
    path: str,
    level=logging.DEBUG,
    **kwargs,
) -> Tuple[Image, Union[str, Iterator[Any]]]:
    resp = client.api.build(path=path, **kwargs)
    if isinstance(resp, str):
        return client.images.get(resp)
    last_event = None
    image_id = None
    result_stream, internal_stream = itertools.tee(json_stream(resp))
    for chunk in internal_stream:
        if "error" in chunk:
            raise BuildError(chunk["error"], result_stream)
        if "stream" in chunk:
            stream = chunk["stream"]
            for line in stream.splitlines():
                logger.log(level, str(line).strip())
            match = re.search(
                r"(^Successfully built |sha256:)([0-9a-f]+)$", stream
            )
            if match:
                image_id = match.group(2)
        last_event = chunk
    if image_id:
        return client.images.get(image_id), result_stream
    raise BuildError(last_event or "Unknown", result_stream)


def _is_docker_running(client: docker.DockerClient) -> bool:
    """
    Check if docker binary and docker daemon are available

    :param client: DockerClient instance
    :return: true or false
    """
    try:
        client.info()
        return True
    except (ImportError, IOError, DockerException):
        return False


def is_docker_running() -> bool:
    """
    Check if docker binary and docker daemon are available

    :return: true or false
    """
    with create_docker_client(check=False) as c:
        return _is_docker_running(c)


_docker_host_lock = Lock()


@contextmanager
def create_docker_client(
    docker_host: str = "", check=True
) -> docker.DockerClient:
    """
    Context manager for DockerClient creation

    :param docker_host: DOCKER_HOST arg for DockerClient
    :param check: check if docker is available
    :return: DockerClient instance
    """
    with _docker_host_lock:
        os.environ[
            "DOCKER_HOST"
        ] = docker_host  # The env var DOCKER_HOST is used to configure docker.from_env()
        client = docker.from_env()
    if check and not _is_docker_running(client):
        raise RuntimeError("Docker daemon is unavailable")
    try:
        yield client
    finally:
        client.close()


def image_exists_at_dockerhub(tag):
    repo, tag = tag.split(":")
    resp = requests.get(
        f"https://registry.hub.docker.com/v1/repositories/{repo}/tags/{tag}"
    )
    time.sleep(1)  # rate limiting
    return resp.status_code == 200


def repository_tags_at_dockerhub(repo):
    resp = requests.get(
        f"https://registry.hub.docker.com/v1/repositories/{repo}/tags"
    )
    time.sleep(1)  # rate limiting
    if resp.status_code != 200:
        return {}
    return {tag["name"] for tag in resp.json()}


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
