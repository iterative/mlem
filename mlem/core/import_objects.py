import pickle
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Type

from mlem.core.artifacts import PlaceholderArtifact, get_file_info
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
    def process(  # pylint: disable=arguments-differ # so what
        cls,
        obj: Location,
        copy_data: bool = True,
        modifier: Optional[str] = None,
        **kwargs
    ) -> MlemMeta:
        raise NotImplementedError

    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        if "type_" in cls.__dict__:
            ImportAnalyzer.types[cls.type_] = cls


class ImportAnalyzer(Analyzer[MlemMeta]):
    base_hook_class = ImportHook
    types: Dict[str, Type[ImportHook]] = {}

    @classmethod
    def analyze(  # pylint: disable=arguments-differ # so what
        cls, obj: Location, copy_data: bool = True, **kwargs
    ) -> MlemMeta:
        return super().analyze(obj, copy_data=copy_data, **kwargs)


class ExtImportHook(ImportHook, ABC):
    EXTS: Tuple[str, ...]

    @classmethod
    def is_object_valid(cls, obj: Location) -> bool:
        return any(obj.path.endswith(e) for e in cls.EXTS)


class PickleImportHook(ExtImportHook):
    EXTS = (".pkl", ".pickle")
    type_ = "pickle"

    @classmethod
    def process(
        cls,
        obj: Location,
        copy_data: bool = True,
        modifier: Optional[str] = None,
        **kwargs
    ) -> MlemMeta:
        with obj.open("rb") as f:
            data = pickle.load(f)
        meta = get_object_metadata(data, **kwargs)
        if not copy_data:
            meta.artifacts = [
                PlaceholderArtifact(
                    location=obj,
                    uri=obj.uri,
                    **get_file_info(obj.fullpath, obj.fs)
                )
            ]
        return meta
