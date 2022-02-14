from pathlib import Path
from typing import List

from click import option
from yaml import safe_dump

from mlem.cli.main import cli, verbose_option
from mlem.config import CONFIG, CONFIG_FILE, MlemConfig
from mlem.constants import MLEM_DIR
from mlem.core.base import build_model
from mlem.utils.root import find_repo_root

# TODO: improve cli for config https://github.com/iterative/mlem/issues/116


@cli.group("config")
def config():
    pass


@config.command("set")
@verbose_option
@option("-c", "--conf", multiple=True)
def config_set(conf: List[str]):
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
