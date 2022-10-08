import os
import platform
import subprocess
import sys
import venv
from abc import abstractmethod
from typing import ClassVar, Optional

from mlem.core.errors import MlemError
from mlem.core.objects import MlemBuilder, MlemModel
from mlem.ui import EMOJI_FAIL, EMOJI_OK, EMOJI_PACK, echo


class EnvBuilder(MlemBuilder):
    """MlemBuilder implementation for building virtual environments"""

    type: ClassVar = "env"

    target: Optional[str] = "venv"
    """Name of the virtual environment"""

    @abstractmethod
    def create_virtual_env(self):
        raise NotImplementedError

    @abstractmethod
    def get_installed_packages(self, env_dir):
        raise NotImplementedError

    @abstractmethod
    def build(self, obj: MlemModel):
        raise NotImplementedError


class VenvBuilder(EnvBuilder):

    type: ClassVar = "venv"

    no_cache: Optional[bool] = False
    """Disable cache"""
    current_env: Optional[bool] = False
    """Whether to install in the current virtual env, must be active"""

    def create_virtual_env(self):
        env_spec = venv.EnvBuilder(with_pip=True)
        env_dir = os.path.abspath(self.target)

        context = env_spec.ensure_directories(env_dir)

        true_system_site_packages = env_spec.system_site_packages
        env_spec.system_site_packages = False
        env_spec.create_configuration(context)
        env_spec.setup_python(context)
        if env_spec.with_pip:
            env_spec._setup_pip(context)  # pylint: disable=protected-access
        if not env_spec.upgrade:
            env_spec.setup_scripts(context)
            env_spec.post_setup(context)
        if true_system_site_packages:
            env_spec.system_site_packages = True
            env_spec.create_configuration(context)

        return context

    def get_installed_packages(self, env_dir):
        if platform.system() == "Windows":
            env_exe = os.path.join(env_dir, "Scripts", "python.exe")
        else:
            env_exe = os.path.join(env_dir, "bin", "python")

        return subprocess.check_output([env_exe, "-m", "pip", "freeze"])

    def build(self, obj: MlemModel):
        if self.current_env:
            if (
                os.getenv("VIRTUAL_ENV") is None
                or sys.prefix == sys.base_prefix
            ):
                raise MlemError("No virtual environment detected.")
            echo(EMOJI_PACK + f"Detected the virtual env {sys.prefix}")
            env_dir = sys.prefix
            env_exe = os.path.join(env_dir, "bin", "python")
        else:
            echo(EMOJI_PACK + f"Creating virtual env {self.target}...")
            context = self.create_virtual_env()
            env_dir = context.env_dir
            os.environ["VIRTUAL_ENV"] = context.env_dir
            env_exe = context.env_exe
        echo(EMOJI_PACK + "Installing the required packages...")
        # Based on recommendation given in https://pip.pypa.io/en/latest/user_guide/#using-pip-from-your-program
        install_cmd = [env_exe, "-m", "pip", "install"]
        if self.no_cache:
            install_cmd.append("--no-cache-dir")
        install_cmd.extend(obj.requirements.to_pip())
        subprocess.run(install_cmd, check=True)
        echo(
            EMOJI_OK
            + f"virtual environment `{self.target}` is ready, activate with `source {self.target}/bin/activate`"
        )
        return env_dir


class CondaBuilder(EnvBuilder):

    type: ClassVar = "conda"

    python_version: str = "3"
    """The python version to use"""
    current_env: Optional[bool] = False
    """Whether to install in the current conda env"""

    def create_virtual_env(self):
        ret = None
        create_cmd = ["--prefix", self.target, f"python={self.python_version}"]
        try:
            ret = subprocess.run(
                ["conda", "create", "-y", *create_cmd], check=True
            )
        except FileNotFoundError:
            echo(
                EMOJI_FAIL
                + f"Couldn't create the conda environment `{self.target}`"
            )
        return ret.returncode

    def get_installed_packages(self, env_dir):
        return subprocess.check_output(["conda", "list", "--prefix", env_dir])

    def build(self, obj: MlemModel):
        pip_based_packages = obj.requirements.to_pip()
        conda_based_packages = obj.requirements.to_conda()

        if self.current_env:
            conda_default_env = os.getenv("CONDA_DEFAULT_ENV", None)
            if conda_default_env == "base" or conda_default_env is None:
                raise MlemError("No conda environment detected.")
            echo(EMOJI_PACK + f"Detected the conda env {sys.prefix}")
            env_dir = sys.prefix
            env_exe = sys.executable
        else:
            assert self.target is not None
            ret = self.create_virtual_env()
            assert ret == 0
            env_dir = self.target
            env_exe = os.path.join(env_dir, "bin", "python")
        try:
            if conda_based_packages:
                ret = subprocess.run(
                    [
                        "conda",
                        "install",
                        "--prefix",
                        env_dir,
                        "-y",
                        *conda_based_packages,
                    ],
                    check=True,
                )
                assert ret.returncode == 0
        except FileNotFoundError:
            echo(
                EMOJI_FAIL
                + f"Couldn't install packages in conda environment `{self.target}`"
            )

        # install pip packages in conda env
        if pip_based_packages:
            subprocess.run(
                [env_exe, "-m", "pip", "install", *pip_based_packages],
                check=True,
            )

        return env_dir
