"""
MLEM's command-line interface
"""
from mlem.cli.apply import apply
from mlem.cli.build import build
from mlem.cli.checkenv import checkenv
from mlem.cli.clone import clone
from mlem.cli.config import config
from mlem.cli.declare import declare
from mlem.cli.deployment import deployment
from mlem.cli.dev import dev
from mlem.cli.import_object import import_object
from mlem.cli.info import pretty_print
from mlem.cli.init import init
from mlem.cli.link import link
from mlem.cli.main import app
from mlem.cli.serve import serve
from mlem.cli.types import list_types

__all__ = [
    "apply",
    "deployment",
    "app",
    "init",
    "build",
    "pretty_print",
    "link",
    "clone",
    "serve",
    "config",
    "declare",
    "import_object",
    "list_types",
    "dev",
    "checkenv",
]


def main():
    app()


if __name__ == "__main__":
    main()
