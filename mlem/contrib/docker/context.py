import glob
import logging
import os
import posixpath
import shutil
import subprocess
import tempfile
from abc import abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, ClassVar, Dict, List, Optional, Union

from fsspec import AbstractFileSystem
from fsspec.implementations.local import LocalFileSystem
from pydantic import BaseModel
from yaml import safe_dump

import mlem
from mlem.config import MlemConfigBase, project_config
from mlem.core.objects import MlemModel
from mlem.core.requirements import Requirements, UnixPackageRequirement
from mlem.runtime.server import Server
from mlem.ui import EMOJI_BUILD, EMOJI_PACK, echo, no_echo
from mlem.utils.importing import import_from_path
from mlem.utils.module import get_python_version
from mlem.utils.templates import TemplateModel

REQUIREMENTS = "requirements.txt"
MLEM_REQUIREMENTS = "mlem_requirements.txt"
SERVER = "server.yaml"
TEMPLATE_FILE = "dockerfile.j2"
MLEM_LOCAL_WHL = f"mlem-{mlem.__version__}-py3-none-any.whl"

logger = logging.getLogger(__name__)


class DockerConfig(MlemConfigBase):
    source: Optional[str] = None
    whl_path: Optional[str] = None
    pip_version: str = mlem.__version__
    git_rev: Optional[str] = None

    class Config:
        env_prefix = "mlem_docker"
        section = "docker"


LOCAL_DOCKER_CONFIG = project_config(None, section=DockerConfig)


class MlemSource:
    type: str
    override = None

    @abstractmethod
    def build(self, path: str):
        raise NotImplementedError

    @abstractmethod
    def get_install_command(self, args: "DockerBuildArgs") -> str:
        raise NotImplementedError

    def __init_subclass__(cls):
        mlem_sources[cls.type] = cls()
        return super().__init_subclass__()


mlem_sources: Dict[str, MlemSource] = {}


class WhlSource(MlemSource):
    type = "whl"

    def build(self, path: str):
        with open(
            os.path.join(path, MLEM_REQUIREMENTS), "w", encoding="utf8"
        ) as f:
            try:
                f.write("\n".join(get_mlem_requirements().to_pip()))
            except FileNotFoundError:
                pass

        config_whl_path = LOCAL_DOCKER_CONFIG.whl_path
        if config_whl_path is not None:
            self._existing_whl(config_whl_path, path)
            return

        self._new_whl(path)

    @staticmethod
    def _new_whl(path):
        mlem_src_path = os.path.dirname(os.path.dirname(mlem.__file__))
        echo(EMOJI_BUILD + "Building MLEM wheel file...")
        logger.debug("Building mlem whl from %s...", mlem_src_path)
        with tempfile.TemporaryDirectory() as whl_dir:
            subprocess.check_output(
                f"pip wheel . --no-deps -w {whl_dir}",
                shell=True,  # nosec: B602
                cwd=mlem_src_path,
            )
            whl_path = glob.glob(os.path.join(whl_dir, "*.whl"))[0]
            shutil.copy(whl_path, os.path.join(path, MLEM_LOCAL_WHL))
        logger.debug("Built mlem whl %s", whl_path)

    @staticmethod
    def _existing_whl(config_whl_path, path):
        if not os.path.isfile(config_whl_path):
            raise ValueError("'whl_path' should be a path to whl file")
        echo(EMOJI_PACK + f"Adding MLEM wheel file from {config_whl_path}")
        shutil.copy(
            config_whl_path,
            os.path.join(path, WhlSource._whl_name()),
        )

    @staticmethod
    def _whl_name():
        if LOCAL_DOCKER_CONFIG.whl_path is not None:
            return os.path.basename(LOCAL_DOCKER_CONFIG.whl_path)
        return MLEM_LOCAL_WHL

    def get_install_command(self, args: "DockerBuildArgs"):
        name = WhlSource._whl_name()
        return (
            f"COPY {MLEM_REQUIREMENTS} .\n"
            f"RUN pip install -r {MLEM_REQUIREMENTS}\n"
            f"COPY {name} .\n"
            f"RUN pip install {name}\n"
        )


class PipSource(MlemSource):
    type = "pip"

    def build(self, path):
        pass

    def get_install_command(self, args: "DockerBuildArgs"):
        return f"RUN pip install mlem=={LOCAL_DOCKER_CONFIG.pip_version}"


class GitSource(MlemSource):
    type = "git"

    def build(self, path):
        pass

    def get_install_command(self, args: "DockerBuildArgs"):
        rev = ""
        if LOCAL_DOCKER_CONFIG.git_rev is not None:
            rev = f"@{LOCAL_DOCKER_CONFIG.git_rev}"
        return (
            f"RUN {args.package_install_cmd} git {args.package_clean_cmd}\n"
            f"RUN pip install git+https://github.com/iterative/mlem{rev}#egg=mlem"
        )


@contextmanager
def use_mlem_source(source="whl"):
    """Context manager that changes docker builder behaviour to copy
    this installation of mlem instead of installing it from pip.
    This is needed for testing and examples"""
    tmp = MlemSource.override
    MlemSource.override = source
    try:
        yield
    finally:
        MlemSource.override = tmp


def get_mlem_source() -> MlemSource:
    source = MlemSource.override or LOCAL_DOCKER_CONFIG.source
    if source in mlem_sources:
        return mlem_sources[source]
    if source is not None:
        raise ValueError(f"unknown mlem source '{source}'")
    # if source is not specified
    if "dev" in mlem.__version__:
        return WhlSource()
    return PipSource()


def get_mlem_requirements():
    requirements = Requirements.new()
    cwd = os.getcwd()
    try:
        setup_mod = import_from_path(
            "setup",
            str(Path(mlem.__file__).parent.parent / "setup.py"),
        )
        setup_args = setup_mod.setup_args

        requirements += list(setup_args["install_requires"])
        logger.debug("Adding MLEM requirements %s", requirements.to_pip())
    finally:
        os.chdir(cwd)
    return requirements


class DockerBuildArgs(BaseModel):
    """Container for DockerBuild arguments"""

    class Config:
        fields = {"prebuild_hook": {"exclude": True}}

    base_image: Optional[Union[str, Callable[[str], str]]] = None
    """base image for the built image in form of a string or function from python version,
    default: python:{python_version}"""
    python_version: str = get_python_version()
    """Python version to use
    default: version of running interpreter"""
    templates_dir: List[str] = []
    """directory or list of directories for Dockerfile templates
       - `pre_install.j2` - Dockerfile commands to run before pip
       - `post_install.j2` - Dockerfile commands to run after pip
       - `post_copy.j2` - Dockerfile commands to run after pip and MLEM distribution copy"""
    run_cmd: Optional[str] = "sh run.sh"
    """command to run in container"""
    package_install_cmd: str = "apt-get update && apt-get -y upgrade && apt-get install --no-install-recommends -y"
    """command to install packages. Default is apt-get, change it for other package manager"""
    package_clean_cmd: str = "&& apt-get clean && rm -rf /var/lib/apt/lists/*"
    """command to clean after package installation"""
    prebuild_hook: Optional[Callable[[str], Any]] = None
    """callable to call before build, accepts python version. Used for pre-building server images"""
    mlem_whl: Optional[str] = None
    """a path to mlem .whl file. If it is empty, mlem will be installed from pip"""
    platform: Optional[str] = None
    """platform to build docker for, see docs.docker.com/desktop/multi-arch/"""

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
    model: MlemModel
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
        envs = {}
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
        self.write_mlem_source()
        self.write_configs()
        self.write_model()
        reqs = self.get_requirements()
        self.write_dockerfile(reqs)
        self.write_local_sources(reqs)
        self.write_requirements_file(reqs)
        self.write_run_file()

    def write_requirements_file(self, requirements: Requirements):
        echo(EMOJI_PACK + "Generating requirements file...")
        with open(
            os.path.join(self.path, REQUIREMENTS), "w", encoding="utf8"
        ) as req:
            logger.debug(
                "Auto-determined requirements for model: %s.",
                requirements.to_pip(),
            )
            req.write("\n".join(requirements.to_pip()))

    def write_model(self):
        echo(EMOJI_PACK + "Adding model files...")
        with no_echo():
            path = os.path.join(self.path, self.model_name)
            if self.model.is_saved:
                self.model.clone(path)
            else:
                copy = self.model.copy()
                copy.model_type.bind(self.model.model_type.model)
                copy.dump(path)

    def write_dockerfile(self, requirements: Requirements):
        echo(EMOJI_BUILD + "Generating dockerfile...")
        env = self.get_env_vars()
        with open(
            os.path.join(self.path, "Dockerfile"), "w", encoding="utf8"
        ) as df:
            unix_packages = requirements.of_type(UnixPackageRequirement)
            dockerfile = DockerfileGenerator(
                **self.docker_args.dict()
            ).generate(
                env=env, packages=[p.package_name for p in unix_packages or []]
            )
            df.write(dockerfile)

    def write_configs(self):
        with self.fs.open(
            posixpath.join(self.path, SERVER), "w", encoding="utf8"
        ) as f:
            safe_dump(self.server.dict(), f)

    def write_local_sources(self, requirements: Requirements):
        echo(EMOJI_PACK + "Adding sources...")
        sources = {}
        for cr in requirements.custom:
            sources.update(cr.to_sources_dict())

        # add __init__.py for all dirs that doesn't have it already
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
            sh.write(f"mlem serve -l {SERVER} -m {self.model_name}")

    def write_mlem_source(self):
        source = get_mlem_source()
        source.build(self.path)


class DockerfileGenerator(DockerBuildArgs, TemplateModel):
    """
    Class to generate Dockerfile
    """

    TEMPLATE_FILE: ClassVar = "dockerfile.j2"
    TEMPLATE_DIR: ClassVar = os.path.dirname(__file__)

    def prepare_dict(self):
        logger.debug(
            'Generating Dockerfile via templates from "%s"...',
            self.templates_dir,
        )

        logger.debug('Docker image is based on "%s".', self.base_image)

        mlem_install = get_mlem_source().get_install_command(self)

        docker_args = {
            "python_version": self.python_version,
            "base_image": self.get_base_image(),
            "run_cmd": self.run_cmd,
            "mlem_install": mlem_install,
            "package_install_cmd": self.package_install_cmd,
            "package_clean_cmd": self.package_clean_cmd,
        }
        return docker_args


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
