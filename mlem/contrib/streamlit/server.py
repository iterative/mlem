import contextlib
import os
import shutil
import subprocess
import tempfile
from threading import Thread
from time import sleep
from typing import ClassVar, Dict, List, Optional

import streamlit
import streamlit_pydantic
from pydantic import BaseModel

from mlem.core.errors import MlemError
from mlem.core.requirements import LibRequirementsMixin, Requirements
from mlem.runtime import Interface
from mlem.runtime.server import Server
from mlem.utils.path import make_posix
from mlem.utils.templates import TemplateModel

TEMPLATE_PY = "_template.py"
SCRIPT_PY = "script.py"


class StreamlitScript(BaseModel):
    server_host: str = "0.0.0.0"
    """Hostname for running FastAPI backend"""
    server_port: int = 8081
    """Port for running FastAPI backend"""
    page_title: str = "MLEM Streamlit UI"
    """Title of the page in browser"""
    title: str = "MLEM Streamlit UI"
    """Title of the page"""
    description: str = ""
    """Additional text after title"""
    args: Dict[str, str] = {}
    """Additional args for custom template"""


class StreamlitTemplate(StreamlitScript, TemplateModel):
    TEMPLATE_FILE: ClassVar = TEMPLATE_PY
    TEMPLATE_DIR: ClassVar = os.path.dirname(__file__)

    def prepare_dict(self):
        d = super().prepare_dict()
        d.update(self.args)
        return d


class StreamlitServer(Server, StreamlitScript, LibRequirementsMixin):
    """Streamlit UI server"""

    type: ClassVar = "streamlit"
    libraries: ClassVar = (streamlit, streamlit_pydantic)
    port_field: ClassVar = "ui_port"

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
    debug: bool = False
    """Update app on template change"""

    def serve(self, interface: Interface):
        with self.prepare_streamlit_script() as path:
            if self.run_server:
                from mlem.contrib.fastapi import FastAPIServer

                if self.debug:
                    Thread(
                        target=lambda: self._idle(path), daemon=True
                    ).start()
                FastAPIServer(
                    host=self.server_host,
                    port=self.server_port,
                    standardize=self.standardize,
                    debug=self.debug,
                ).serve(interface)
            else:
                while True:
                    self._idle(path)

    def _idle(self, path):
        while True:
            sleep(1)
            if self.debug:
                self._write_streamlit_script(path)

    def _write_streamlit_script(self, path):
        templates_dir = []
        dirname = os.path.dirname(path)
        if self.template is not None:
            shutil.copy(self.template, os.path.join(dirname, TEMPLATE_PY))
            templates_dir = [dirname]
        StreamlitTemplate.from_model(
            self,
            templates_dir=templates_dir,
        ).write(path)

    @contextlib.contextmanager
    def prepare_streamlit_script(self):
        with tempfile.TemporaryDirectory(
            prefix="mlem_streamlit_script_"
        ) as tempdir:
            path = os.path.join(tempdir, SCRIPT_PY)
            self._write_streamlit_script(path)
            with self.run_streamlit_daemon(path):
                yield path

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

    def get_ports(self) -> List[int]:
        return [self.ui_port, self.server_port]
