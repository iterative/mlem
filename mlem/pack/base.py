from abc import abstractmethod
from typing import ClassVar

from mlem.core.base import MlemObject
from mlem.core.objects import ModelMeta


class Packager(MlemObject):
    """"""

    class Config:
        type_root = True

    abs_name: ClassVar[str] = "packager"

    @abstractmethod
    def package(
        self, obj: ModelMeta, out: str
    ):  # TODO maybe we can also pack datasets?
        raise NotImplementedError
