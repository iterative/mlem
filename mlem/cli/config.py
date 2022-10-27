import posixpath
from typing import Optional

from typer import Argument, Option, Typer
from yaml import safe_dump, safe_load

from mlem.cli.main import app, mlem_command, mlem_group, option_project
from mlem.config import get_config_cls
from mlem.constants import MLEM_CONFIG_FILE_NAME
from mlem.core.base import SmartSplitDict, get_recursively, smart_split
from mlem.core.errors import MlemError
from mlem.core.meta_io import get_fs, get_uri
from mlem.ui import EMOJI_OK, echo
from mlem.utils.root import find_project_root

config = Typer(name="config", cls=mlem_group("common"))
app.add_typer(config)


@config.callback()
def config_callback():
    """Manipulate MLEM configuration."""


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

    Documentation: <https://mlem.ai/doc/command-reference/config>
    """
    fs, path = get_fs(project or "")
    project = find_project_root(path, fs=fs)
    try:
        section, name = name.split(".", maxsplit=1)
    except ValueError as e:
        raise MlemError("[name] should contain at least one dot") from e
    config_file_path = posixpath.join(project, MLEM_CONFIG_FILE_NAME)
    with fs.open(config_file_path) as f:
        new_conf = safe_load(f) or {}

    conf = SmartSplitDict(new_conf.get(section, {}))
    conf[name] = value
    new_conf[section] = conf.build()
    if validate:
        config_cls = get_config_cls(section)
        config_cls(**new_conf[section])
    with fs.open(config_file_path, "w", encoding="utf8") as f:
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

    Documentation: <https://mlem.ai/doc/command-reference/config>
    """
    fs, path = get_fs(project or "")
    project = find_project_root(path, fs=fs)
    with fs.open(posixpath.join(project, MLEM_CONFIG_FILE_NAME)) as f:
        try:
            echo(get_recursively(safe_load(f), smart_split(name, ".")))
        except KeyError as e:
            raise MlemError(f"No such option `{name}`") from e
