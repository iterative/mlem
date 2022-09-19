from typing import Optional

from mlem.config import MlemConfigBase


class AWSConfig(MlemConfigBase):
    ROLE: Optional[str]
    PROFILE: Optional[str]

    class Config:
        section = "aws"
        env_prefix = "AWS_"
