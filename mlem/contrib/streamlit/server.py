import contextlib
import os
import shutil
import subprocess
import tempfile
from time import sleep
from typing import ClassVar, Dict, Optional

import streamlit
import streamlit_pydantic

from mlem.core.errors import MlemError
from mlem.core.requirements import LibRequirementsMixin, Requirements
from mlem.runtime import Interface
from mlem.runtime.server import Server
from mlem.utils.path import make_posix
from mlem.utils.templates import TemplateModel

TEMPLATE_PY = "_template.py"
SCRIPT_PY = "script.py"


class StreamlitScript(TemplateModel):
    TEMPLATE_FILE: ClassVar = TEMPLATE_PY
    TEMPLATE_DIR: ClassVar = os.path.dirname(__file__)

    server_host: str = "0.0.0.0"
    server_port: str = "8080"


class StreamlitServer(Server, LibRequirementsMixin):
    """Streamlit UI server"""

    type: ClassVar = "streamlit"
    libraries: ClassVar = (streamlit, streamlit_pydantic)

    server_host: str = "0.0.0.0"
    """Hostname for running FastAPI backend"""
    server_port: int = 8080
    """Port for running FastAPI backend"""
    run_server: bool = True
    """Whether to run backend server or use existing one"""
    ui_host: str = "0.0.0.0"
    """Hostname for running Streamlit UI"""
    ui_port: int = 80
    """Port for running Streamlit UI"""
    use_watchdog: bool = True
    """Install watchdog for better performance"""
    template: Optional[str] = None
    """Path to alternative template for streamlit app"""
    standardize: bool = False  # changing default for streamlit
    """Use standard model interface"""

    def serve(self, interface: Interface):
        with self.prepare_streamlit_script():
            if self.run_server:
                from mlem.contrib.fastapi import FastAPIServer

                FastAPIServer(
                    host=self.server_host,
                    port=self.server_port,
                    standardize=self.standardize,
                ).serve(interface)
            else:
                while True:
                    sleep(100)

    @contextlib.contextmanager
    def prepare_streamlit_script(self):
        with tempfile.TemporaryDirectory(
            prefix="mlem_streamlit_script_"
        ) as tempdir:
            path = os.path.join(tempdir, SCRIPT_PY)
            templates_dir = []
            if self.template is not None:
                shutil.copy(self.template, os.path.join(tempdir, TEMPLATE_PY))
                templates_dir = [tempdir]
            StreamlitScript(
                server_host=self.server_host,
                server_port=str(self.server_port),
                templates_dir=templates_dir,
            ).write(path)
            with self.run_streamlit_daemon(path):
                yield

    @contextlib.contextmanager
    def run_streamlit_daemon(self, path):
        with subprocess.Popen(  # nosec B603
            [
                "streamlit",
                "run",
                "--server.headless",
                "true",
                "--server.address",
                self.ui_host,
                "--server.port",
                str(self.ui_port),
                path,
            ],
            close_fds=True,
        ):
            yield

    def get_requirements(self) -> Requirements:
        reqs = super().get_requirements()
        if self.run_server:
            from mlem.contrib.fastapi import FastAPIServer

            reqs += FastAPIServer().get_requirements()
        if self.use_watchdog:
            reqs += Requirements.new("watchdog")
        return reqs

    def get_sources(self) -> Dict[str, bytes]:
        sources = super().get_sources()
        if self.template is not None:
            if not os.path.abspath(self.template).startswith(
                os.path.abspath(".")
            ):
                # TODO: issue#...
                raise MlemError(
                    f"Template file {self.template} is not in a subtree of cwd"
                )
            template = make_posix(
                os.path.relpath(os.path.abspath(self.template), ".")
            )
            with open(template, "rb") as f:
                sources[template] = f.read()
            self.template = template
        return sources
