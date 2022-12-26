import contextlib
import os
import subprocess
import tempfile
from time import sleep
from typing import Callable, ClassVar, Optional, Tuple, Type

import streamlit
import streamlit_pydantic
from pydantic import BaseModel

from mlem.cli.utils import LIST_LIKE_SHAPES
from mlem.core.requirements import LibRequirementsMixin, Requirements
from mlem.runtime import Interface
from mlem.runtime.server import Server
from mlem.utils.templates import TemplateModel

SCRIPT_PY = "script.py"


class StreamlitScript(TemplateModel):
    TEMPLATE_FILE: ClassVar = "_template.py"
    TEMPLATE_DIR: ClassVar = os.path.dirname(__file__)

    method_name: str
    server_host: str = "0.0.0.0"
    server_port: str = "8080"


def augment_model(
    model: Type,
) -> Tuple[Callable, Optional[Type]]:
    if not issubclass(model, BaseModel):
        return lambda x: x, model
    list_model = [
        (name, f)
        for name, f in model.__fields__.items()
        if f.shape in LIST_LIKE_SHAPES
    ]

    if len(list_model) == 0:
        return lambda x: x, model

    if len(list_model) > 1:
        return lambda x: x, None
    name, field = list_model[0]

    return lambda x: model(**{name: [x]}), field.type_


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

    def serve(self, interface: Interface):
        with self.prepare_streamlit_script(interface):
            if self.run_server:
                from mlem.contrib.fastapi import FastAPIServer

                FastAPIServer(
                    host=self.server_host, port=self.server_port
                ).serve(interface)
            else:
                while True:
                    sleep(100)

    @contextlib.contextmanager
    def prepare_streamlit_script(self, interface: Interface):
        with tempfile.TemporaryDirectory(
            prefix="mlem_streamlit_script_"
        ) as tempdir:
            path = os.path.join(tempdir, SCRIPT_PY)
            StreamlitScript(
                server_host=self.server_host,
                server_port=str(self.server_port),
                method_name=interface.get_method_names()[0],
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
