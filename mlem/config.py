"""
Configuration management for MLEM
"""
import posixpath
from typing import Any, Dict, List, Optional

import yaml
from fsspec import AbstractFileSystem
from fsspec.implementations.local import LocalFileSystem
from pydantic import BaseSettings, Extra, Field, parse_obj_as
from pydantic.env_settings import InitSettingsSource

from mlem.constants import MLEM_DIR

CONFIG_FILE_NAME = "config.yaml"


def _set_location_init_source(init_source: InitSettingsSource):
    def inner(settings: "MlemConfig"):
        for arg in ("config_path", "config_fs"):
            if arg in init_source.init_kwargs:
                settings.__dict__[arg] = init_source.init_kwargs[arg]
        return {}

    return inner


def mlem_config_settings_source(section: Optional[str]):
    """
    A simple settings source that loads variables from a section of a yaml file in MLEM DIR
    """

    def inner(settings: BaseSettings) -> Dict[str, Any]:
        from mlem.utils.root import find_repo_root

        encoding = settings.__config__.env_file_encoding
        fs = getattr(settings, "config_fs", LocalFileSystem())
        config_path = getattr(settings, "config_path", "")
        repo = find_repo_root(config_path, fs=fs, raise_on_missing=False)
        if repo is None:
            return {}
        config_file = posixpath.join(repo, MLEM_DIR, CONFIG_FILE_NAME)
        if not fs.exists(config_file):
            return {}
        with fs.open(config_file, encoding=encoding) as f:
            conf = yaml.safe_load(f)
        if conf and section:
            conf = conf.get(section, {})
        return {k.upper(): v for k, v in conf.items()} if conf else {}

    return inner


class MlemConfigBase(BaseSettings):
    config_path: str = ""
    config_fs: Optional[AbstractFileSystem] = None

    class Config:
        env_prefix = "mlem_"
        env_file_encoding = "utf-8"
        section: Optional[str] = None
        extra = Extra.allow

        @classmethod
        def customise_sources(
            cls,
            init_settings,
            env_settings,
            file_secret_settings,
        ):
            return (
                _set_location_init_source(init_settings),
                init_settings,
                env_settings,
                mlem_config_settings_source(cls.section),
                file_secret_settings,
            )


class MlemConfig(MlemConfigBase):
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
    EMOJIS: bool = True

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
        return (
            self.ADDITIONAL_EXTENSIONS_RAW.split(  # pylint: disable=no-member
                ","
            )
        )


CONFIG = MlemConfig()


def repo_config(
    repo: str, fs: Optional[AbstractFileSystem] = None
) -> MlemConfig:
    if fs is None:
        fs = LocalFileSystem()
    return MlemConfig(config_path=repo, config_fs=fs)
