import os
import platform
import subprocess
import sys
import venv
from abc import abstractmethod
from typing import ClassVar, List, Optional

from mlem.contrib.requirements import CondaPackageRequirement
from mlem.core.errors import MlemError
from mlem.core.objects import MlemBuilder, MlemModel
from mlem.ui import EMOJI_OK, EMOJI_PACK, echo


def get_python_exe_in_virtual_env(env_dir: str, use_conda_env: bool = False):
    if platform.system() == "Windows":
        if not use_conda_env:
            return os.path.join(env_dir, "Scripts", "python.exe")
        return os.path.join(env_dir, "python.exe")
    return os.path.join(env_dir, "bin", "python")


def run_in_subprocess(cmd: List[str], error_msg: str, check_output=False):
    try:
        if check_output:
            return subprocess.check_output(cmd)
        return subprocess.run(cmd, check=True)
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ) as e:
        raise MlemError(f"{error_msg}\n{e}") from e


class EnvBuilder(MlemBuilder):
    """MlemBuilder implementation for building virtual environments"""

    type: ClassVar = "env"

    target: Optional[str] = "venv"
    """Name of the virtual environment"""

    @abstractmethod
    def create_virtual_env(self):
        raise NotImplementedError

    @abstractmethod
    def get_installed_packages(self, env_dir: str):
        raise NotImplementedError


class VenvBuilder(EnvBuilder):
    type: ClassVar = "venv"

    no_cache: bool = False
    """Disable cache"""
    current_env: bool = False
    """Whether to install in the current virtual env, must be active"""

    def create_virtual_env(self):
        env_dir = os.path.abspath(self.target)
        venv.create(env_dir, with_pip=True)

    def get_installed_packages(self, env_dir):
        env_exe = get_python_exe_in_virtual_env(env_dir)
        return run_in_subprocess(
            [env_exe, "-m", "pip", "freeze"],
            error_msg="Error running pip",
            check_output=True,
        )

    def build(self, obj: MlemModel):
        if self.current_env:
            if (
                os.getenv("VIRTUAL_ENV") is None
                or sys.prefix == sys.base_prefix
            ):
                raise MlemError("No virtual environment detected.")
            echo(EMOJI_PACK + f"Detected the virtual env {sys.prefix}")
            env_dir = sys.prefix
        else:
            assert self.target is not None
            echo(EMOJI_PACK + f"Creating virtual env {self.target}...")
            self.create_virtual_env()
            env_dir = os.path.abspath(self.target)
            os.environ["VIRTUAL_ENV"] = env_dir

        env_exe = get_python_exe_in_virtual_env(env_dir)
        echo(EMOJI_PACK + "Installing the required packages...")
        # Based on recommendation given in https://pip.pypa.io/en/latest/user_guide/#using-pip-from-your-program
        install_cmd = [env_exe, "-m", "pip", "install"]
        if self.no_cache:
            install_cmd.append("--no-cache-dir")
        install_cmd.extend(obj.requirements.to_pip())
        run_in_subprocess(install_cmd, error_msg="Error running pip")
        if platform.system() == "Windows":
            activate_cmd = f"`{self.target}\\Scripts\\activate`"
        else:
            activate_cmd = f"`source {self.target}/bin/activate`"
        echo(
            EMOJI_OK
            + f"virtual environment `{self.target}` is ready, activate with {activate_cmd}"
        )
        return env_dir


class CondaBuilder(EnvBuilder):

    type: ClassVar = "conda"

    python_version: str = f"{sys.version_info.major}.{sys.version_info.minor}"
    """The python version to use"""
    current_env: Optional[bool] = False
    """Whether to install in the current conda env"""
    conda_reqs: List[CondaPackageRequirement] = []
    """List of conda package requirements"""

    def create_virtual_env(self):
        env_dir = os.path.abspath(self.target)
        create_cmd = ["--prefix", env_dir, f"python={self.python_version}"]
        run_in_subprocess(
            ["conda", "create", "-y", *create_cmd],
            error_msg="Error running conda",
        )

    def get_installed_packages(self, env_dir):
        return run_in_subprocess(
            ["conda", "list", "--prefix", env_dir],
            error_msg="Error running conda",
            check_output=True,
        )

    def build(self, obj: MlemModel):  # pylint: disable=too-many-branches
        pip_based_packages = obj.requirements.to_pip()
        conda_based_packages = [r.get_repr() for r in self.conda_reqs]

        if self.current_env:
            conda_default_env = os.getenv("CONDA_DEFAULT_ENV", None)
            if conda_default_env == "base" or conda_default_env is None:
                raise MlemError("No conda environment detected.")
            echo(EMOJI_PACK + f"Detected the conda env {sys.prefix}")
            env_dir = sys.prefix
            env_exe = sys.executable
        else:
            assert self.target is not None
            self.create_virtual_env()
            env_dir = os.path.abspath(self.target)
            env_exe = get_python_exe_in_virtual_env(
                env_dir, use_conda_env=True
            )
        if conda_based_packages:
            run_in_subprocess(
                [
                    "conda",
                    "install",
                    "--prefix",
                    env_dir,
                    "-y",
                    *conda_based_packages,
                ],
                error_msg="Error running conda",
            )

        # install pip packages in conda env
        if pip_based_packages:
            run_in_subprocess(
                [env_exe, "-m", "pip", "install", *pip_based_packages],
                error_msg="Error running pip",
            )

        return env_dir
