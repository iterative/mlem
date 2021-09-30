from abc import ABC

from mlem.core.base import MlemObject


class BaseClient(MlemObject, ABC):
    """TODO: https://github.com/iterative/mlem/issues/40"""


class HTTPClient(BaseClient):
    """TODO: https://github.com/iterative/mlem/issues/40"""

    host: str
    port: int
