import os
from abc import ABC, abstractmethod
from typing import ClassVar, Dict, List, Optional

from mlem.core.base import MlemObject
from mlem.core.requirements import WithRequirements
from mlem.runtime.interface.base import Interface


class Server(MlemObject, ABC, WithRequirements):
    class Config:
        type_root = True

    abs_name: ClassVar[str] = "server"
    env_vars: ClassVar[Optional[Dict[str, str]]] = None
    additional_source_files: ClassVar[Optional[List[str]]] = None

    @abstractmethod
    def serve(self, interface: Interface):
        raise NotImplementedError()

    def get_env_vars(self) -> Dict[str, str]:
        return self.env_vars or {}

    def get_sources(self) -> Dict[str, bytes]:
        res = {}
        for path in self.additional_source_files or []:
            with open(path, "rb") as f:
                res[os.path.basename(path)] = f.read()
        return res
