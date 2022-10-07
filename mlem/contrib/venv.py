import os
import subprocess
import venv
from typing import ClassVar, Optional

from mlem.core.objects import MlemBuilder, MlemModel
from mlem.ui import EMOJI_OK, EMOJI_PACK, echo


class VenvBuilder(MlemBuilder):
    """MlemBuilder implementation for building virtual environments"""

    type: ClassVar = "venv"

    target: Optional[str] = "venv"
    """Name of the virtual environment"""
    no_cache: Optional[bool] = False
    """Disable cache"""

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

    def get_installed_packages(self, context):
        return subprocess.check_output(
            [context.env_exe, "-m", "pip", "freeze"]
        )

    def build(self, obj: MlemModel):
        echo(EMOJI_PACK + f"Creating virtual env {self.target}...")
        context = self.create_virtual_env()
        os.environ["VIRTUAL_ENV"] = context.env_dir
        echo(EMOJI_PACK + "Installing the required packages...")
        # Based on recommendation given in https://pip.pypa.io/en/latest/user_guide/#using-pip-from-your-program
        install_cmd = [context.env_exe, "-m", "pip", "install"]
        if self.no_cache:
            install_cmd.append("--no-cache-dir")
        install_cmd.extend(obj.requirements.to_pip())
        subprocess.run(install_cmd, check=True)
        echo(
            EMOJI_OK
            + f"virtual environment `{self.target}` is ready, activate with `source {self.target}/bin/activate`"
        )
        return context
