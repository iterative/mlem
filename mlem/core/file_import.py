import pickle
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Type

from mlem.core.hooks import Analyzer, Hook
from mlem.core.meta_io import Location
from mlem.core.metadata import get_object_metadata
from mlem.core.objects import MlemMeta


class ImportHook(Hook[MlemMeta], ABC):
    type_: str

    @classmethod
    @abstractmethod
    def is_object_valid(cls, obj: Location) -> bool:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def process(cls, obj: Location, **kwargs) -> MlemMeta:
        raise NotImplementedError

    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        if "type_" in cls.__dict__:
            ImportAnalyzer.types[cls.type_] = cls


class ImportAnalyzer(Analyzer[MlemMeta]):
    base_hook_class = ImportHook
    types: Dict[str, Type[ImportHook]] = {}

    @classmethod
    def analyze(cls, obj: Location, **kwargs) -> MlemMeta:
        return super().analyze(obj, **kwargs)


class ExtImportHook(ImportHook, ABC):
    EXTS: Tuple[str, ...]

    @classmethod
    def is_object_valid(cls, obj: Location) -> bool:
        return any(obj.path.endswith(e) for e in cls.EXTS)


class PickleImportHook(ExtImportHook):
    EXTS = (".pkl", ".pickle")
    type_ = "pickle"

    @classmethod
    def process(cls, obj: Location, **kwargs) -> MlemMeta:
        with obj.open("rb") as f:
            data = pickle.load(f)
        return get_object_metadata(data, **kwargs)
