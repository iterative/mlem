import pickle
from abc import ABC, abstractmethod
from typing import ClassVar, Optional, Tuple, Union

from mlem.core.artifacts import PlaceholderArtifact, get_file_info
from mlem.core.base import MlemABC
from mlem.core.errors import FileNotFoundOnImportError
from mlem.core.hooks import Analyzer, Hook
from mlem.core.meta_io import Location
from mlem.core.metadata import get_object_metadata
from mlem.core.model import ModelIO
from mlem.core.objects import MlemData, MlemModel, MlemObject


class ImportHook(Hook[MlemObject], MlemABC, ABC):
    """Base class for defining import hooks.
    On every import attempt all available hooks are checked if the imported path
    represented by `Location` instance if valid for them. Then process method is
    called on a hook that first passed the check"""

    type: ClassVar[str]
    abs_name: ClassVar = "import"

    class Config:
        type_root = True

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
        **kwargs,
    ) -> MlemObject:
        raise NotImplementedError


class ImportAnalyzer(Analyzer[MlemObject]):
    base_hook_class: ClassVar = ImportHook

    @classmethod
    def analyze(  # pylint: disable=arguments-differ # so what
        cls, obj: Location, copy_data: bool = True, **kwargs
    ) -> MlemObject:
        if not obj.exists():
            raise FileNotFoundOnImportError(f"Nothing found at {obj.uri}")
        return super().analyze(obj, copy_data=copy_data, **kwargs)


class ExtImportHook(ImportHook, ABC):
    """Base class for import hooks that target particular file extensions"""

    EXTS: ClassVar[Tuple[str, ...]]

    @classmethod
    def is_object_valid(cls, obj: Location) -> bool:
        return any(obj.path.endswith(e) for e in cls.EXTS)


class LoadAndAnalyzeImportHook(ImportHook, ABC):
    force_type: ClassVar[Union[MlemModel, MlemData, None]] = None

    @classmethod
    def process(
        cls,
        obj: Location,
        copy_data: bool = True,
        modifier: Optional[str] = None,
        **kwargs,
    ) -> MlemObject:
        data = cls.load_obj(obj, modifier, **kwargs)
        if cls.force_type is None:
            meta = get_object_metadata(data, **kwargs)
        elif cls.force_type is MlemModel:
            meta = MlemModel.from_obj(data, **kwargs)
        else:
            meta = MlemData.from_data(data, **kwargs)
        if not copy_data:
            meta.artifacts = {
                ModelIO.art_name: PlaceholderArtifact(
                    location=obj,
                    uri=obj.uri,
                    **get_file_info(obj.fullpath, obj.fs),
                )
            }
        return meta

    @classmethod
    def load_obj(cls, location: Location, modifier: Optional[str], **kwargs):
        raise NotImplementedError


class PickleImportHook(ExtImportHook, LoadAndAnalyzeImportHook):
    """Import hook for pickle files"""

    EXTS: ClassVar = (".pkl", ".pickle")
    type: ClassVar = "pickle"

    @classmethod
    def load_obj(cls, location: Location, modifier: Optional[str], **kwargs):
        with location.open("rb") as f:
            return pickle.load(f)
