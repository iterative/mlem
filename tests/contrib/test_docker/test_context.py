import os
import tempfile

import docker

import mlem
from mlem.contrib.docker.base import DockerImage, DockerIORegistry
from mlem.contrib.docker.context import DockerfileGenerator, use_mlem_source
from mlem.core.requirements import UnixPackageRequirement
from tests.contrib.test_docker.conftest import docker_test

REGISTRY_PORT = 5000


def test_dockerfile_generator_custom_python_version():
    dockerfile = _cut_empty_lines(
        f"""FROM python:AAAA-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install mlem=={mlem.__version__}
COPY . ./
CMD sh run.sh
"""
    )

    kwargs = {"python_version": "AAAA"}
    assert _generate_dockerfile(**kwargs) == dockerfile


def test_dockerfile_generator_unix_packages():
    dockerfile = _cut_empty_lines(
        f"""FROM python:3.6-slim
WORKDIR /app
RUN kek aaa bbb
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install mlem=={mlem.__version__}
COPY . ./
CMD sh run.sh
"""
    )

    kwargs = {"python_version": "3.6", "package_install_cmd": "kek"}
    assert (
        _generate_dockerfile(
            **kwargs,
            unix_packages=[
                UnixPackageRequirement(package_name="aaa"),
                UnixPackageRequirement(package_name="bbb"),
            ],
        )
        == dockerfile
    )


def test_dockerfile_generator_super_custom():
    dockerfile = _cut_empty_lines(
        f"""FROM my-python:3.6
WORKDIR /app
RUN echo "pre_install"
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install mlem=={mlem.__version__}
RUN echo "post_install"
COPY . ./
RUN echo "post_copy"
CMD echo "cmd" && sh run.sh
"""
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        for hook in ("pre_install", "post_install", "post_copy"):
            with open(
                os.path.join(tmpdir, f"{hook}.j2"), "w", encoding="utf8"
            ) as f:
                f.write(f'RUN echo "{hook}"')

        kwargs = {
            "base_image": "my-python:3.6",
            "templates_dir": [tmpdir],
            "run_cmd": 'echo "cmd" && sh run.sh',
        }
        assert _generate_dockerfile(**kwargs) == dockerfile


def test_use_wheel_installation(tmpdir):
    distr = tmpdir.mkdir("distr").join("somewhatwheel.txt")
    distr.write("wheel goes brrr")
    with use_mlem_source(str(distr)):
        dockerfile = DockerfileGenerator(
            mlem_whl=str(os.path.basename(distr))
        ).generate(env={}, packages=[])
        assert "RUN pip install somewhatwheel.txt" in dockerfile


def _cut_empty_lines(string):
    return "\n".join(line for line in string.splitlines() if line)


def _generate_dockerfile(unix_packages=None, **kwargs):
    with use_mlem_source():
        return _cut_empty_lines(
            DockerfileGenerator(**kwargs).generate(
                env={}, packages=[p.package_name for p in unix_packages or []]
            )
        )


@docker_test
def test_docker_registry_io():
    registry = DockerIORegistry()
    client = docker.DockerClient()

    client.images.pull("hello-world:latest")

    assert registry.get_host() == "https://index.docker.io/v1/"
    registry.push(client, "hello-world:latest")
    image = DockerImage(name="hello-world")
    assert registry.image_exists(client, image)


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
