import posixpath
from typing import Optional

from typer import Argument, Typer
from yaml import safe_dump, safe_load

from mlem.cli.main import MlemGroupSection, app, mlem_command, option_repo
from mlem.config import CONFIG_FILE_NAME
from mlem.constants import MLEM_DIR
from mlem.core.base import get_recursively, set_recursively, smart_split
from mlem.core.errors import MlemError
from mlem.core.meta_io import get_fs
from mlem.ui import EMOJI_OK, echo
from mlem.utils.root import find_repo_root

config = Typer(name="config", cls=MlemGroupSection("common"))
app.add_typer(config)


@config.callback()
def config_callback():
    """Manipulate MLEM configuration"""


@mlem_command("set", parent=config)
def config_set(
    name: str = Argument(..., help="Dotted name of option"),
    value: str = Argument(..., help="New value"),
    repo: Optional[str] = option_repo,
):
    """Set configuration value

    Examples:
        $ mlem config set pandas.default_format csv
    """
    fs, path = get_fs(repo or "")
    repo = find_repo_root(path, fs=fs)
    with fs.open(posixpath.join(repo, MLEM_DIR, CONFIG_FILE_NAME)) as f:
        new_conf = safe_load(f) or {}
    set_recursively(new_conf, smart_split(name, "."), value)
    config_file = posixpath.join(repo, MLEM_DIR, CONFIG_FILE_NAME)
    with fs.open(config_file, "w", encoding="utf8") as f:
        safe_dump(
            new_conf,
            f,
        )
    echo(EMOJI_OK + f"Set `{name}` to `{value}` in repo {repo}")


@mlem_command("get", parent=config)
def config_get(
    name: str = Argument(..., help="Dotted name of option"),
    repo: Optional[str] = option_repo,
):
    """Get configuration value

    Examples:
        $ mlem config get pandas.default_format
        $ mlem config get pandas.default_format --repo https://github.com/iterative/example-mlem/
    """
    fs, path = get_fs(repo or "")
    repo = find_repo_root(path, fs=fs)
    with fs.open(posixpath.join(repo, MLEM_DIR, CONFIG_FILE_NAME)) as f:
        try:
            echo(get_recursively(safe_load(f), smart_split(name, ".")))
        except KeyError as e:
            raise MlemError(f"No such option `{name}`") from e
