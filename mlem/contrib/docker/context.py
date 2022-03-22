import glob
import logging
import os
import posixpath
import shutil
import tempfile
from contextlib import contextmanager
from typing import Any, Callable, ClassVar, Dict, List, Optional, Union

from fsspec import AbstractFileSystem
from fsspec.implementations.local import LocalFileSystem
from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel

from mlem.core.objects import ModelMeta
from mlem.core.requirements import Requirements, UnixPackageRequirement
from mlem.runtime.server.base import Server
from mlem.utils.module import get_python_version

REQUIREMENTS = "requirements.txt"
TEMPLATE_FILE = "dockerfile.j2"

logger = logging.getLogger(__name__)

_MLEM_SOURCE = "pip"
MLEM_INSTALL_COMMAND = "pip install mlem=={version}"


def mlem_from_pip():
    """
    :return boolen flag if mlem inside image must be installed from pip (or copied local dist instread)"""
    return _MLEM_SOURCE == "pip"


def mlem_from_local():
    return _MLEM_SOURCE == "local"


def mlem_from_whl():
    return not mlem_from_pip() and not mlem_from_local()


@contextmanager
def use_mlem_source(source="local"):
    """Context manager that changes docker builder behaviour to copy
    this installation of mlem instead of installing it from pip.
    This is needed for testing and examples"""
    global _MLEM_SOURCE  # pylint: disable=global-statement  # TODO is there a better way?
    tmp = _MLEM_SOURCE
    _MLEM_SOURCE = source
    try:
        yield
    finally:
        _MLEM_SOURCE = tmp


class DockerBuildArgs(BaseModel):
    """
    Container for DockerBuild arguments

    :param base_image:  base image for the built image in form of a string or function from python version,
      default: python:{python_version}
    :param python_version: Python version to use, default: version of running interpreter
    :param templates_dir: directory or list of directories for Dockerfile templates, default: ./docker_templates
       - `pre_install.j2` - Dockerfile commands to run before pip
       - `post_install.j2` - Dockerfile commands to run after pip
       - `post_copy.j2` - Dockerfile commands to run after pip and MLEM distribution copy
    :param run_cmd: command to run in container, default: sh run.sh
    :param package_install_cmd: command to install packages. Default is apt-get, change it for other package manager
    :param prebuild_hook: callable to call before build, accepts python version. Used for pre-building server images
    :param mlem_whl: a path to mlem .whl file. If it is empty, mlem will be installed from pip TODO
    :param platform: platform to build docker for, see https://docs.docker.com/desktop/multi-arch/
    """

    base_image: Optional[Union[str, Callable[[str], str]]] = None
    python_version: str = get_python_version()
    templates_dir: List[str] = []
    run_cmd: Union[bool, str] = "sh run.sh"
    package_install_cmd: str = "apt-get install -y"
    prebuild_hook: Optional[Callable[[str], Any]] = None
    mlem_whl: Optional[str] = None
    platform: Optional[str] = None

    def get_base_image(self):
        if self.base_image is None:
            return f"python:{self.python_version}-slim"
        if isinstance(self.base_image, str):
            return self.base_image
        if not callable(self.base_image):
            raise ValueError(f"Invalid value {self.base_image} for base_image")
        return self.base_image(  # pylint: disable=not-callable # but it is
            self.python_version
        )

    def update(self, other: "DockerBuildArgs"):
        for (
            field
        ) in (
            DockerBuildArgs.__fields_set__  # pylint: disable=not-an-iterable # dunno, it is Set[str]
        ):
            if field == "templates_dir":
                self.templates_dir += other.templates_dir
            else:
                value = getattr(other, field)
                if value is not None:
                    setattr(self, field, value)


class DockerModelDirectory(BaseModel):
    model: ModelMeta
    server: Server
    docker_args: "DockerBuildArgs"
    debug: bool
    path: str
    model_name: str = "model"

    fs: ClassVar[
        AbstractFileSystem
    ] = (
        LocalFileSystem()
    )  # TODO: https://github.com/iterative/mlem/issues/38 fs

    def get_requirements(self) -> Requirements:
        return (
            self.model.requirements + self.server.get_requirements()
        )  # TODO: tmp

    def get_env_vars(self) -> Dict[str, str]:
        """Get env variables for image"""
        envs = {
            # LOADER_ENV: self.loader.classpath,
            # SERVER_ENV: self.server.classpath,
            # 'MLEM_RUNTIME': 'true'
        }
        if self.debug:
            envs["MLEM_DEBUG"] = "true"

        envs.update(self.server.get_env_vars())

        modules = set(self.get_requirements().modules)

        from mlem.ext import ExtensionLoader

        extensions = ExtensionLoader.loaded_extensions().keys()
        used_extensions = [
            e.module for e in extensions if all(r in modules for r in e.reqs)
        ]
        if len(used_extensions) > 0:
            envs["MLEM_EXTENSIONS"] = ",".join(used_extensions)
        return envs

    def get_python_version(self):
        """Returns current python version"""
        return get_python_version()

    def write_distribution(self):
        logger.debug('Writing model distribution to "%s"...', self.path)
        os.makedirs(self.path, exist_ok=True)
        self.write_mlem_whl()
        self.write_configs()
        self.write_model()
        reqs = self.get_requirements()
        self.write_dockerfile(reqs)
        self.write_local_sources(reqs)
        self.write_requirements_file(reqs)
        self.write_run_file()

    def write_requirements_file(self, requirements: Requirements):
        with open(
            os.path.join(self.path, REQUIREMENTS), "w", encoding="utf8"
        ) as req:
            logger.debug(
                "Auto-determined requirements for model: %s.",
                requirements.to_pip(),
            )
            if (
                mlem_from_pip() is False
            ):  # TODO: https://github.com/iterative/mlem/issues/39
                cwd = os.getcwd()
                try:
                    from setup import setup_args  # only for development

                    requirements += list(setup_args["install_requires"])
                    logger.debug(
                        "Adding MLEM requirements as local installation is employed..."
                    )
                    logger.debug(
                        "Overall requirements for model: %s.",
                        requirements.to_pip(),
                    )
                finally:
                    os.chdir(cwd)
            req.write("\n".join(requirements.to_pip()))

    def write_model(self):
        path = os.path.join(self.path, self.model_name)
        if self.model.is_saved:
            self.model.clone(path)
        else:
            self.model.copy().dump(path)

    def write_dockerfile(self, requirements: Requirements):
        env = self.get_env_vars()
        with open(
            os.path.join(self.path, "Dockerfile"), "w", encoding="utf8"
        ) as df:
            unix_packages = requirements.of_type(UnixPackageRequirement)
            dockerfile = _DockerfileGenerator(self.docker_args).generate(
                env, unix_packages
            )
            df.write(dockerfile)

    def write_configs(self):
        pass

    def write_local_sources(self, requirements: Requirements):
        sources = {}
        for cr in requirements.custom:
            sources.update(cr.to_sources_dict())

        # add __init__.py for all dirs that doesnt have it already
        packages = {
            os.path.join(os.path.dirname(p), "__init__.py")
            for p in sources
            if os.path.dirname(p) != ""
        }
        sources.update({p: "" for p in packages if p not in sources})
        sources.update(self.server.get_sources())
        for path, src in sources.items():
            logger.debug('Putting model source "%s" to distribution...', path)
            full_path = posixpath.join(self.path, path)
            self.fs.makedirs(posixpath.dirname(full_path), exist_ok=True)
            with self.fs.open(full_path, "wb") as f:
                f.write(src)

    def write_run_file(self):
        with self.fs.open(posixpath.join(self.path, "run.sh"), "w") as sh:
            sh.write(f"mlem serve {self.model_name} {self.server.type}")

    def write_mlem_whl(self):

        if mlem_from_pip():
            # nothing to do
            return
        if mlem_from_whl():
            # set whl option
            logger.debug(
                "Putting MLEM wheel to distribution as wheel installation is employed..."
            )
            shutil.copy(
                _MLEM_SOURCE,
                os.path.join(self.path, os.path.basename(_MLEM_SOURCE)),
            )
            self.docker_args.mlem_whl = _MLEM_SOURCE
            return
        # build whl and set option
        import subprocess

        import mlem

        repo_path = os.path.dirname(os.path.dirname(mlem.__file__))
        logger.debug("Build mlem whl from %s...", repo_path)
        with tempfile.TemporaryDirectory() as whl_dir:
            subprocess.check_output(
                f"cd {repo_path} && python setup.py bdist_wheel -d {whl_dir}",
                shell=True,
            )
            whl_path = glob.glob(os.path.join(whl_dir, "*.whl"))[0]
            whl_name = os.path.basename(whl_path)
            shutil.copy(whl_path, os.path.join(self.path, whl_name))
        logger.debug("Built mlem whl %s", whl_path)
        self.docker_args.mlem_whl = whl_name


class _DockerfileGenerator:
    """
    Class to generate Dockerfile

    :param args: DockerBuildArgs instance
    """

    def __init__(self, args: DockerBuildArgs):
        self.python_version = args.python_version
        self.base_image = args.get_base_image()
        self.templates_dir = args.templates_dir
        self.run_cmd = args.run_cmd
        self.package_install_cmd = args.package_install_cmd
        self.mlem_whl = args.mlem_whl

    def generate(
        self,
        env: Dict[str, str],
        packages: List[UnixPackageRequirement] = None,
    ):
        """Generate Dockerfile using provided base image, python version and run_cmd

        :param env: dict with environmental variables
        :param packages: list of unix packages to install
        """
        logger.debug(
            'Generating Dockerfile via templates from "%s"...',
            self.templates_dir,
        )
        j2 = Environment(
            loader=FileSystemLoader(
                [os.path.dirname(__file__)] + self.templates_dir
            )
        )
        docker_tmpl = j2.get_template(TEMPLATE_FILE)

        logger.debug(
            "Docker image is using Python version: %s.", self.python_version
        )
        logger.debug('Docker image is based on "%s".', self.base_image)

        is_whl = self.mlem_whl is not None
        import mlem

        docker_args = {
            "python_version": self.python_version,
            "base_image": self.base_image,
            "run_cmd": self.run_cmd,
            "mlem_install": f"COPY {self.mlem_whl} .\nRUN pip install {self.mlem_whl}"
            if is_whl
            else "RUN "
            + MLEM_INSTALL_COMMAND.format(version=mlem.__version__),
            "env": env,
            "package_install_cmd": self.package_install_cmd,
            "packages": [p.package_name for p in packages or []],
        }

        return docker_tmpl.render(**docker_args)


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
