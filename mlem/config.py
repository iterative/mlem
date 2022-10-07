"""
Configuration management for MLEM
"""
import posixpath
from typing import Any, Dict, List, Optional, Type, TypeVar, Union, overload

import yaml
from fsspec import AbstractFileSystem
from fsspec.implementations.local import LocalFileSystem
from pydantic import BaseSettings, Field, parse_obj_as, root_validator
from pydantic.env_settings import InitSettingsSource

from mlem.constants import MLEM_CONFIG_FILE_NAME
from mlem.core.errors import UnknownConfigSection
from mlem.utils.entrypoints import MLEM_CONFIG_ENTRY_POINT, load_entrypoints


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
        from mlem.utils.root import find_project_root

        encoding = settings.__config__.env_file_encoding
        fs = getattr(settings, "config_fs", LocalFileSystem())
        config_path = getattr(settings, "config_path", "")
        project = find_project_root(config_path, fs=fs, raise_on_missing=False)
        if project is None:
            return {}
        config_file = posixpath.join(project, MLEM_CONFIG_FILE_NAME)
        if not fs.exists(config_file):
            return {}
        with fs.open(config_file, encoding=encoding) as f:
            conf = yaml.safe_load(f)
        if conf and section:
            conf = conf.get(section, {})
        return {k.upper(): v for k, v in conf.items()} if conf else {}

    return inner


T = TypeVar("T", bound="MlemConfigBase")


class MlemConfigBase(BaseSettings):
    """Special base for mlem settings to be able to read them from files"""

    config_path: str = ""
    config_fs: Optional[AbstractFileSystem] = None

    class Config:
        env_prefix = "mlem_"
        env_file_encoding = "utf-8"
        section: Optional[str] = None

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

    @root_validator(pre=True)
    def ignore_case(
        cls, value: Any
    ):  # pylint: disable=no-self-argument  # noqa: B902
        new_value = {}
        if isinstance(value, dict):
            for key, val in value.items():
                if key.upper() in cls.__fields__:
                    key = key.upper()
                if key.lower() in cls.__fields__:
                    key = key.lower()
                new_value[key] = val
        return new_value

    @classmethod
    def local(cls: Type[T]) -> T:
        return project_config("", section=cls)


class MlemConfig(MlemConfigBase):
    """Base Mlem Config"""

    class Config:
        section = "core"

    GITHUB_USERNAME: Optional[str] = Field(default=None, env="GITHUB_USERNAME")
    GITHUB_TOKEN: Optional[str] = Field(default=None, env="GITHUB_TOKEN")
    ADDITIONAL_EXTENSIONS: str = Field(default="")
    AUTOLOAD_EXTS: bool = True
    LOG_LEVEL: str = "INFO"
    DEBUG: bool = False
    NO_ANALYTICS: bool = False
    TESTS: bool = False
    STORAGE: Dict = {}
    EMOJIS: bool = True
    STATE: Dict = {}
    SERVER: Dict = {}

    @property
    def storage(self):
        from mlem.core.artifacts import LOCAL_STORAGE, Storage

        if not self.STORAGE:
            return LOCAL_STORAGE
        s = parse_obj_as(Storage, self.STORAGE)
        return s

    @property
    def additional_extensions(self) -> List[str]:
        if self.ADDITIONAL_EXTENSIONS == "":
            return []
        return self.ADDITIONAL_EXTENSIONS.split(  # pylint: disable=no-member
            ","
        )

    @property
    def state(self):
        if not self.STATE:
            return None
        from mlem.core.objects import StateManager

        return parse_obj_as(StateManager, self.STATE)

    @property
    def server(self):
        from mlem.runtime.server import Server

        if not self.SERVER:
            return parse_obj_as(Server, {"type": "fastapi"})
        return parse_obj_as(Server, self.SERVER)


LOCAL_CONFIG = MlemConfig()


def get_config_cls(section: str) -> Type[MlemConfigBase]:
    try:
        return load_entrypoints(MLEM_CONFIG_ENTRY_POINT)[section].ep.load()
    except KeyError as e:
        raise UnknownConfigSection(section) from e


@overload
def project_config(
    project: Optional[str],
    fs: Optional[AbstractFileSystem] = None,
    section: Type[MlemConfig] = MlemConfig,
) -> MlemConfig:
    ...


@overload
def project_config(
    project: Optional[str],
    fs: Optional[AbstractFileSystem] = None,
    section: str = ...,
) -> MlemConfigBase:
    ...


@overload
def project_config(
    project: Optional[str],
    fs: Optional[AbstractFileSystem] = None,
    section: Type[T] = ...,
) -> T:
    ...


def project_config(
    project: Optional[str],
    fs: Optional[AbstractFileSystem] = None,
    section: Union[Type[MlemConfigBase], str] = MlemConfig,
) -> MlemConfigBase:
    if isinstance(section, str):
        cls = get_config_cls(section)
    else:
        cls = section
    if fs is None:
        fs = LocalFileSystem()
    return cls(config_path=project or "", config_fs=fs)
