"""
MLEM's command-line interface
"""
from mlem.cli.apply import apply
from mlem.cli.clone import clone
from mlem.cli.config import config
from mlem.cli.create import create
from mlem.cli.deploy import deploy
from mlem.cli.import_path import import_path
from mlem.cli.info import ls, pretty_print
from mlem.cli.init import init
from mlem.cli.link import link
from mlem.cli.main import cli
from mlem.cli.package import pack
from mlem.cli.serve import serve

__all__ = [
    "apply",
    "deploy",
    "cli",
    "init",
    "pack",
    "pretty_print",
    "link",
    "ls",
    "clone",
    "serve",
    "config",
    "create",
    "import_path",
]


def main():
    cli()


if __name__ == "__main__":
    main()
