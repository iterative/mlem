import contextlib
import os
import subprocess
import tempfile
from typing import ClassVar

from mlem.contrib.fastapi import FastAPIServer
from mlem.runtime import Interface
from mlem.runtime.server import Server
from mlem.utils.templates import TemplateModel

SCRIPT_PY = "script.py"


class StreamlitScript(TemplateModel):
    TEMPLATE_FILE: ClassVar = "template.py"
    TEMPLATE_DIR: ClassVar = os.path.dirname(__file__)

    method_name: str


@contextlib.contextmanager
def run_streamlit_daemon(path):
    with subprocess.Popen(  # nosec B603
        ["streamlit", "run", "--server.headless", "true", path],
        close_fds=True,
    ):
        yield


@contextlib.contextmanager
def prepare_streamlit_script(interface: Interface):
    with tempfile.TemporaryDirectory(
        prefix="mlem_streamlit_script_"
    ) as tempdir:
        path = os.path.join(tempdir, SCRIPT_PY)
        StreamlitScript(method_name=interface.get_method_names()[0]).write(
            path
        )
        with run_streamlit_daemon(path):
            yield


class StreamlitServer(Server):
    type: ClassVar = "streamlit"

    def serve(self, interface: Interface):
        with prepare_streamlit_script(interface):
            FastAPIServer().serve(interface)
