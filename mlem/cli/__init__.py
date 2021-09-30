"""
MLEM's command-line interface
"""
from mlem.cli.apply import apply
from mlem.cli.deploy import deploy
from mlem.cli.env import environment
from mlem.cli.get import get
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
    "environment",
    "init",
    "pack",
    "pretty_print",
    "link",
    "ls",
    "get",
    "serve",
]


def main():
    cli()


if __name__ == "__main__":
    main()
