import glob
import logging
import os.path
import posixpath
import tempfile
from typing import ClassVar, Dict, List, Optional

from fsspec import AbstractFileSystem
from fsspec.implementations.local import LocalFileSystem

import mlem
from mlem.core.meta_io import get_fs
from mlem.core.objects import ModelMeta
from mlem.core.requirements import InstallableRequirement
from mlem.pack import Packager
from mlem.utils.module import get_python_version
from mlem.utils.templates import TemplateModel

logger = logging.getLogger(__name__)


class SetupTemplate(TemplateModel):
    TEMPLATE_FILE: ClassVar = "setup.py.j2"
    TEMPLATE_DIR: ClassVar = os.path.dirname(__file__)

    package_name: str
    python_version: Optional[str] = None
    short_description: str = ""
    url: str = ""
    email: str = ""
    author: str = ""
    version: str = "0.0.0"
    additional_setup_kwargs: Dict = {}


class SourceTemplate(TemplateModel):
    TEMPLATE_FILE: ClassVar = "source.py.j2"
    TEMPLATE_DIR: ClassVar = os.path.dirname(__file__)

    methods: List[str]


class PipMixin(SetupTemplate):
    def prepare_dict(self):
        self.python_version = self.python_version or get_python_version()
        return SetupTemplate.dict(self, include=set(SetupTemplate.__fields__))

    def make_distr(self, obj: ModelMeta, root: str, fs: AbstractFileSystem):
        path = posixpath.join(root, self.package_name)
        fs.makedirs(path, exist_ok=True)
        self.write(posixpath.join(root, "setup.py"), fs)

        # TODO: methods with correct signatures
        SourceTemplate(methods=list(obj.model_type.methods)).write(
            posixpath.join(path, "__init__.py"), fs
        )

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


class PipPackager(Packager, PipMixin):
    type: ClassVar = "pip"
    target: str

    def package(self, obj: ModelMeta):
        fs, root = get_fs(self.target)
        self.make_distr(obj, root, fs)


class WhlPackager(Packager, PipMixin):
    type: ClassVar = "whl"
    target: str

    def build_whl(self, path, target, target_fs):
        import subprocess

        target_fs.makedirs(target, exist_ok=True)
        logger.debug("Building whl from %s...", path)
        with tempfile.TemporaryDirectory() as whl_dir:
            subprocess.check_output(
                f"cd {path} && pip wheel . --no-deps -w {whl_dir}",
                shell=True,
            )
            whl_path = glob.glob(os.path.join(whl_dir, "*.whl"))[0]
            whl_name = os.path.basename(whl_path)

            target_fs.upload(whl_path, posixpath.join(target, whl_name))

    def package(self, obj: ModelMeta):
        fs, path = get_fs(self.target)
        with tempfile.TemporaryDirectory() as tmpdir:
            self.make_distr(obj, str(tmpdir), LocalFileSystem())
            self.build_whl(tmpdir, path, fs)
