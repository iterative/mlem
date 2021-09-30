from abc import ABC

from mlem.core.base import MlemObject


class BaseClient(MlemObject, ABC):
    """TODO client class"""


class HTTPClient(BaseClient):
    """TODO"""

    host: str
    port: int
