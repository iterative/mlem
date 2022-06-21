import posixpath
from typing import Optional

from typer import Argument, Option, Typer
from yaml import safe_dump, safe_load

from mlem.cli.main import MlemGroupSection, app, mlem_command, option_project
from mlem.config import CONFIG_FILE_NAME, get_config_cls
from mlem.constants import MLEM_DIR
from mlem.core.base import get_recursively, set_recursively, smart_split
from mlem.core.errors import MlemError
from mlem.core.meta_io import get_fs, get_uri
from mlem.ui import EMOJI_OK, echo
from mlem.utils.root import find_project_root

config = Typer(name="config", cls=MlemGroupSection("common"))
app.add_typer(config)


@config.callback()
def config_callback():
    """Manipulate MLEM configuration"""


@mlem_command("set", parent=config)
def config_set(
    name: str = Argument(..., help="Dotted name of option"),
    value: str = Argument(..., help="New value"),
    project: Optional[str] = option_project,
    validate: bool = Option(
        True, help="Whether to validate config schema after"
    ),
):
    """Set configuration value

    Examples:
        $ mlem config set pandas.default_format csv
    """
    fs, path = get_fs(project or "")
    project = find_project_root(path, fs=fs)
    try:
        section, name = name.split(".", maxsplit=1)
    except ValueError as e:
        raise MlemError("[name] should contain at least one dot") from e
    with fs.open(posixpath.join(project, MLEM_DIR, CONFIG_FILE_NAME)) as f:
        new_conf = safe_load(f) or {}

    new_conf[section] = new_conf.get(section, {})
    set_recursively(new_conf[section], smart_split(name, "."), value)
    if validate:
        config_cls = get_config_cls(section)
        config_cls(**new_conf[section])
    config_file = posixpath.join(project, MLEM_DIR, CONFIG_FILE_NAME)
    with fs.open(config_file, "w", encoding="utf8") as f:
        safe_dump(
            new_conf,
            f,
        )
    echo(
        EMOJI_OK
        + f"Set `{name}` to `{value}` in project {get_uri(fs, path, True)}"
    )


@mlem_command("get", parent=config)
def config_get(
    name: str = Argument(..., help="Dotted name of option"),
    project: Optional[str] = option_project,
):
    """Get configuration value

    Examples:
        $ mlem config get pandas.default_format
        $ mlem config get pandas.default_format --project https://github.com/iterative/example-mlem/
    """
    fs, path = get_fs(project or "")
    project = find_project_root(path, fs=fs)
    with fs.open(posixpath.join(project, MLEM_DIR, CONFIG_FILE_NAME)) as f:
        try:
            echo(get_recursively(safe_load(f), smart_split(name, ".")))
        except KeyError as e:
            raise MlemError(f"No such option `{name}`") from e
