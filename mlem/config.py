from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseSettings, Field

CONFIG_FILE = "config.yaml"


def mlem_config_settings_source(settings: BaseSettings) -> Dict[str, Any]:
    """
    A simple settings source that loads variables from a yaml file in MLEM DIR
    """
    from mlem.utils.root import find_mlem_root

    encoding = settings.__config__.env_file_encoding
    mlem_root = find_mlem_root(raise_on_missing=False)
    if mlem_root is None:
        return {}
    config_file = Path(mlem_root) / CONFIG_FILE
    if not config_file.exists():
        return {}
    return yaml.safe_load((config_file).read_text(encoding))


class MlemConfig(BaseSettings):
    GITHUB_USERNAME: Optional[str]
    GITHUB_TOKEN: Optional[str]
    ADDITIONAL_EXTENSIONS_RAW: str = Field(
        default="", env="MLEM_ADDITIONAL_EXTENSIONS"
    )
    AUTOLOAD_EXTS: bool = True
    DEFAULT_BRANCH: str = "main"

    @property
    def ADDITIONAL_EXTENSIONS(self) -> List[str]:
        if self.ADDITIONAL_EXTENSIONS_RAW == "":
            return []
        else:
            return self.ADDITIONAL_EXTENSIONS_RAW.split(",")

    class Config:
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
