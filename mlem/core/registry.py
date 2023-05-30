from abc import ABC
from typing import ClassVar

from mlem.core.base import MlemABC


class ArtifactInRegistry(MlemABC, ABC):
    """Artifact registered within an Artifact Registry.
    Can provide version and stage it's promoted to.
    """

    class Config:
        type_root = True
        default_type = "gto"

    abs_name: ClassVar = "artifact_in_registry"
    uri: str
    """location"""

    @property
    def version(self):
        raise NotImplementedError

    @property
    def stage(self):
        raise NotImplementedError
