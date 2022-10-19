import glob
import logging
import os.path
import posixpath
import subprocess
import tempfile
from typing import Any, ClassVar, Dict, List, Optional

from fsspec import AbstractFileSystem
from fsspec.implementations.local import LocalFileSystem
from pydantic import validator

import mlem
from mlem.core.meta_io import get_fs, get_uri
from mlem.core.objects import MlemBuilder, MlemModel
from mlem.core.requirements import InstallableRequirement
from mlem.ui import EMOJI_PACK, echo, no_echo
from mlem.utils.module import get_python_version
from mlem.utils.templates import TemplateModel

logger = logging.getLogger(__name__)


class SetupTemplate(TemplateModel):
    TEMPLATE_FILE: ClassVar = "setup.py.j2"
    TEMPLATE_DIR: ClassVar = os.path.dirname(__file__)

    package_name: str
    """Name of python package"""
    python_version: Optional[str] = None
    """Required python version"""
    short_description: str = ""
    """short_description"""
    url: str = ""
    """url"""
    email: str = ""
    """author's email"""
    author: str = ""
    """author's name"""
    version: str = "0.0.0"
    """package version"""
    additional_setup_kwargs: Dict[str, Any] = {}
    """additional parameters for setup()"""

    @validator("python_version")
    def validate_python_version(  # pylint: disable=no-self-argument
        cls, value  # noqa: B902
    ):
        return f"=={value}" if value and value[0] in "0123456789" else value


class SourceTemplate(TemplateModel):
    TEMPLATE_FILE: ClassVar = "source.py.j2"
    TEMPLATE_DIR: ClassVar = os.path.dirname(__file__)

    methods: List[str]
    """list of methods"""


class PipMixin(SetupTemplate):
    def prepare_dict(self):
        self.python_version = (
            self.python_version or f"=={get_python_version()}"
        )
        return SetupTemplate.dict(self, include=set(SetupTemplate.__fields__))

    def make_distr(self, obj: MlemModel, root: str, fs: AbstractFileSystem):
        path = posixpath.join(root, self.package_name)
        fs.makedirs(path, exist_ok=True)
        self.write(posixpath.join(root, "setup.py"), fs)

        # TODO: methods with correct signatures
        SourceTemplate(methods=list(obj.model_type_raw["methods"])).write(
            posixpath.join(path, "__init__.py"), fs
        )
        with no_echo():
            obj.clone(posixpath.join(path, "model"), fs)
        with fs.open(posixpath.join(root, "requirements.txt"), "w") as f:
            f.write(
                "\n".join(
                    (
                        obj.requirements
                        + InstallableRequirement.from_module(mlem)
                    ).to_pip()
                )
            )
        with fs.open(posixpath.join(root, "MANIFEST.in"), "w") as f:
            f.write(f"graft {self.package_name}")
        echo(
            EMOJI_PACK
            + f"Written `{self.package_name}` package data to `{get_uri(fs, root, True)}`"
        )


class PipBuilder(MlemBuilder, PipMixin):
    """Create a directory python package"""

    type: ClassVar = "pip"
    target: str
    """Path to save result"""

    def build(self, obj: MlemModel):
        fs, root = get_fs(self.target)
        self.make_distr(obj, root, fs)


class WhlBuilder(MlemBuilder, PipMixin):
    """Create a wheel with python package"""

    type: ClassVar = "whl"
    target: str
    """Path to save result"""

    def build_whl(self, path, target, target_fs):
        target_fs.makedirs(target, exist_ok=True)
        logger.debug("Building whl from %s...", path)
        with tempfile.TemporaryDirectory() as whl_dir:
            subprocess.check_output(
                f"pip wheel . --no-deps -w {whl_dir}",
                shell=True,  # nosec: B602
                cwd=path,
            )
            whl_path = glob.glob(os.path.join(whl_dir, "*.whl"))[0]
            whl_name = os.path.basename(whl_path)

            target_fs.upload(whl_path, posixpath.join(target, whl_name))

    def build(self, obj: MlemModel):
        fs, path = get_fs(self.target)
        with tempfile.TemporaryDirectory() as tmpdir:
            self.make_distr(obj, str(tmpdir), LocalFileSystem())
            self.build_whl(tmpdir, path, fs)
