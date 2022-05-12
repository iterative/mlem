from abc import abstractmethod
from typing import ClassVar

from mlem.core.objects import MlemModel, MlemObject


class Packager(MlemObject):
    """"""

    class Config:
        type_root = True
        type_field = "type"

    object_type: ClassVar = "packager"
    abs_name: ClassVar[str] = "packager"

    @abstractmethod
    def package(self, obj: MlemModel):  # TODO maybe we can also pack datasets?
        raise NotImplementedError
