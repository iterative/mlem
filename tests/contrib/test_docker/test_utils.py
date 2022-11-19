import pytest

from mlem.contrib.docker.utils import (
    image_exists_at_dockerhub,
    repository_tags_at_dockerhub,
)
from mlem.utils.module import get_python_version
from tests.contrib.test_docker.conftest import docker_test


@docker_test
def test_image_exists():
    assert image_exists_at_dockerhub(
        f"python:{get_python_version()}-slim", library=True
    )
    assert image_exists_at_dockerhub("minio/minio:latest")
    assert image_exists_at_dockerhub("postgres:alpine", library=True)
    assert image_exists_at_dockerhub("registry:latest", library=True)


@docker_test
def test_image_not_exists():
    assert not image_exists_at_dockerhub("python:this_does_not_exist")
    assert not image_exists_at_dockerhub("mlem:this_does_not_exist")
    assert not image_exists_at_dockerhub("minio:this_does_not_exist")
    assert not image_exists_at_dockerhub("registry:this_does_not_exist")
    assert not image_exists_at_dockerhub("this_does_not_exist:latest")


@docker_test
def test_repository_tags(request):
    tags = repository_tags_at_dockerhub("python", library=True)
    python_version = get_python_version()
    if python_version == "3.8.14":
        request.applymarker(pytest.mark.xfail)
    assert f"{python_version}-slim" in tags
    assert python_version in tags

    tags = repository_tags_at_dockerhub("minio/minio")
    assert "latest" in tags


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
