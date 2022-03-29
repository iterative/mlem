from pathlib import Path
from typing import List

from typer import Option, Typer
from yaml import safe_dump

from mlem.cli.main import mlem_command
from mlem.config import CONFIG, CONFIG_FILE, MlemConfig
from mlem.constants import MLEM_DIR
from mlem.core.base import build_model
from mlem.utils.root import find_repo_root

# TODO: improve cli for config https://github.com/iterative/mlem/issues/116

config = Typer()


@config.callback()
def config_callback():
    pass


@mlem_command("set", parent=config)
def config_set(
    conf: List[str] = Option(
        ...,
        "-c",
        "--conf",
    )
):
    repo = find_repo_root()
    new_conf = build_model(MlemConfig, conf, [], **CONFIG.dict())
    config_file = Path(repo) / MLEM_DIR / CONFIG_FILE
    with open(config_file, "w", encoding="utf8") as f:
        safe_dump(
            new_conf.dict(
                exclude_unset=True, exclude_defaults=True, by_alias=True
            ),
            f,
        )
