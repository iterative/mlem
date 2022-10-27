from typing import ClassVar, List

from fsspec import AbstractFileSystem
from fsspec.implementations.local import LocalFileSystem
from jinja2 import (
    Environment,
    FileSystemLoader,
    StrictUndefined,
    select_autoescape,
)
from pydantic import BaseModel


class TemplateModel(BaseModel):
    """Base class to render jinja templates from pydantic models"""

    TEMPLATE_FILE: ClassVar[str]
    TEMPLATE_DIR: ClassVar[str]

    templates_dir: List[str] = []
    """list of directories to look for jinja templates"""

    def prepare_dict(self):
        return self.dict()

    def generate(self, **additional):
        j2 = Environment(
            loader=FileSystemLoader(self.templates_dir + [self.TEMPLATE_DIR]),
            undefined=StrictUndefined,
            autoescape=select_autoescape(),
        )
        template = j2.get_template(self.TEMPLATE_FILE)
        args = self.prepare_dict()
        args.update(additional)
        return template.render(**args)

    def write(self, path: str, fs: AbstractFileSystem = None, **additional):
        fs = fs or LocalFileSystem()
        with fs.open(path, "w") as f:
            f.write(self.generate(**additional))
