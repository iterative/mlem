"""
Configuration management for MLEM
"""
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseSettings, Field, parse_obj_as

from mlem.constants import MLEM_DIR

CONFIG_FILE = "config.yaml"


def mlem_config_settings_source(settings: BaseSettings) -> Dict[str, Any]:
    """
    A simple settings source that loads variables from a yaml file in MLEM DIR
    """
    from mlem.utils.root import find_repo_root

    encoding = settings.__config__.env_file_encoding
    repo = find_repo_root(raise_on_missing=False)
    if repo is None:
        return {}
    config_file = Path(repo) / MLEM_DIR / CONFIG_FILE
    if not config_file.exists():
        return {}
    conf = yaml.safe_load(config_file.read_text(encoding=encoding))

    return {k.upper(): v for k, v in conf.items()} if conf else {}


class MlemConfig(BaseSettings):
    GITHUB_USERNAME: Optional[str] = Field(default=None, env="GITHUB_USERNAME")
    GITHUB_TOKEN: Optional[str] = Field(default=None, env="GITHUB_TOKEN")
    ADDITIONAL_EXTENSIONS_RAW: str = Field(
        default="", env="MLEM_ADDITIONAL_EXTENSIONS"
    )
    AUTOLOAD_EXTS: bool = True
    DEFAULT_BRANCH: str = "main"
    LOG_LEVEL: str = "INFO"
    DEBUG: bool = False
    NO_ANALYTICS: bool = False
    TESTS: bool = False
    DEFAULT_STORAGE: Dict = {}
    DEFAULT_EXTERNAL: bool = False

    # class Config:
    #     extra = Extra.allow

    @property
    def default_storage(self):
        from mlem.core.artifacts import LOCAL_STORAGE, Storage

        if not self.DEFAULT_STORAGE:
            return LOCAL_STORAGE
        s = parse_obj_as(Storage, self.DEFAULT_STORAGE)
        return s

    @property
    def ADDITIONAL_EXTENSIONS(self) -> List[str]:
        if self.ADDITIONAL_EXTENSIONS_RAW == "":
            return []
        return self.ADDITIONAL_EXTENSIONS_RAW.split(",")

    class Config:
        env_prefix = "mlem_"
        env_file_encoding = "utf-8"

        @classmethod
        def customise_sources(
            cls,
            init_settings,
            env_settings,
            file_secret_settings,
        ):
            return (
                init_settings,
                env_settings,
                mlem_config_settings_source,
                file_secret_settings,
            )


CONFIG = MlemConfig()
