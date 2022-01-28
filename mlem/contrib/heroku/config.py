from typing import Optional

from mlem.config import MlemConfigBase


class HerokuConfig(MlemConfigBase):
    API_KEY: Optional[str] = None

    class Config:
        env_prefix = "heroku_"
        section = "heroku"


HEROKU_CONFIG = HerokuConfig()
