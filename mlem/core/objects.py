"""
Base classes for meta objects in MLEM:
MlemMeta and it's subclasses, e.g. ModelMeta, DatasetMeta, etc
"""
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Any, ClassVar, Dict, Optional, Tuple, Type, TypeVar, Union

from fsspec import AbstractFileSystem
from fsspec.implementations.local import LocalFileSystem
from yaml import safe_dump, safe_load

from mlem.config import CONFIG
from mlem.core.artifacts import Artifacts, FSSpecStorage
from mlem.core.base import MlemObject
from mlem.core.dataset_type import Dataset, DatasetReader
from mlem.core.errors import (
    MlemError,
    MlemObjectNotSavedError,
    MlemRootNotFound,
    ObjectExistsError,
)
from mlem.core.meta_io import (
    ART_DIR,
    META_FILE_NAME,
    MLEM_DIR,
    MLEM_EXT,
    deserialize,
    get_fs,
    resolve_fs,
    serialize,
)
from mlem.core.model import ModelAnalyzer, ModelType
from mlem.core.requirements import Requirements
from mlem.polydantic.lazy import lazy_field
from mlem.utils.root import find_mlem_root


class Deployment(MlemObject):
    __type_root__ = True
    abs_name: ClassVar[str] = "deployment"

    # @abstractmethod
    # def get_client(self) -> BaseClient:
    #     pass

    @abstractmethod
    def get_status(self):
        pass

    @abstractmethod
    def destroy(self):
        pass


T = TypeVar("T", bound="MlemMeta")


class MlemMeta(MlemObject):
    __type_root__ = True
    abs_name: ClassVar[str] = "meta"
    __type_field__ = "object_type"
    __transient_fields__ = {"name", "fs"}
    object_type: ClassVar[str]
    _path: Optional[str] = None
    _fs: ClassVar[Optional[AbstractFileSystem]] = None

    @classmethod
    def _get_location(cls, path:str, fs: AbstractFileSystem = None, mlem_root:str=None, ensure_mlem_root: bool = True):
        fullpath = os.path.abspath(os.path.join(mlem_root or "", path))
        if fs is None:
            fs, fullpath = get_fs(fullpath)
        if ensure_mlem_root:
            if mlem_root is None:
                mlem_root = find_mlem_root(fullpath, fs)
            else:
                mlem_root = find_mlem_root(mlem_root, fs)
        else:
            mlem_root = None
        return fs, fullpath, mlem_root

    def bind(self, name: str, fs: AbstractFileSystem):
        self._path = name
        self._fs = fs
        return self

    @classmethod
    def read(
        cls: Type[T],
        path: str,
        fs: AbstractFileSystem = None,
        mlem_root: str = None,
        *,
        follow_links: bool = True,
    ) -> T:
        """
        Read object in (path, fs)
            and try to deserialise it as `cls` instance

        Args:
            path: Exact path to MLEM metafile,
            fs: Filesystem on which path is located,
            follow_links: If deserialised object is a MLEM link,
                whether to load and return the linked object
                or just return MlemLink object.

        Returns:
            Deserialised object
        """
        fs, path, _ = cls._get_location(path, fs, mlem_root)
        with fs.open(path) as f:
            payload = safe_load(f)
        res = deserialize(payload, cls).bind(path, fs)
        if follow_links and isinstance(res, MlemLink):
            return res.load_link()
        return res

    def write_value(
        self, mlem_root: str  # pylint: disable=unused-argument
    ) -> Optional[Artifacts]:
        return None

    def load_value(self):
        pass

    def unload_value(self):
        pass

    def get_value(self):
        return self


    def dump(self, path: str, fs: Union[str, AbstractFileSystem] = None, mlem_root: Optional[str] = None,
             link: bool = True, check_extension: bool = True, absolute: bool = False):
        """Dumps metafile and possible artifacts to path."""
        fs, fullpath, mlem_root = self._get_location(path, fs, mlem_root, ensure_mlem_root=link)
        if link or mlem_root is None:
            if check_extension and not path.endswith(MLEM_EXT):
                raise ValueError(f"name={path} should end with {MLEM_EXT}")
            path = os.path.join(mlem_root or "", path)
        else:
            path = mlem_dir_path(
                path,
                fs=fs,
                mlem_root=mlem_root,
                obj_type=self.object_type,
            )
        self._write_meta(fs, link, mlem_root, path)

    def _write_meta(self, fs, link, mlem_root, path):
        fs.makedirs(os.path.dirname(path), exist_ok=True)
        with fs.open(path, "w") as f:
            safe_dump(serialize(self), f)
        if link:
            self.make_link_in_mlem_dir(mlem_root)

    def make_link(
        self,
        path: str = None,
        fs: AbstractFileSystem = None,
        raise_on_exist: bool = False,
        mlem_root: str = None,
    ) -> "MlemLink":
        if self.name is None:
            raise MlemObjectNotSavedError(
                "Cannot create link for not saved meta object"
            )
        if path is not None:
            mlem_root = mlem_root or find_mlem_root(path, fs, False) or ""
            mlem_link = os.path.relpath(self.name, mlem_root)
        else:
            mlem_link = self.name
        link = MlemLink(mlem_link=mlem_link, link_type=self.object_type)
        if path is not None:
            if raise_on_exist and os.path.exists(path):
                raise ObjectExistsError(f"Object at {path} already exists")
            link.dump(path, fs=fs, absolute=True)
        return link

    def make_link_in_mlem_dir(self, mlem_root: str = None) -> None:
        if self.name is None:
            raise MlemObjectNotSavedError(
                "Cannot create link for not saved meta object"
            )
        path = mlem_dir_path(
            os.path.dirname(self.name),
            fs=self.fs,
            obj_type=self.object_type,
            mlem_root=mlem_root,
        )
        self.make_link(path=path, fs=self.fs, mlem_root=mlem_root)

    @classmethod
    def subtype_mapping(cls) -> Dict[str, Type["MlemMeta"]]:
        return cls.__type_map__

    # def get_artifacts(self) -> ArtifactCollection:
    #     return Blobs({})

    def clone(
        self,
        name: str,
        how: str = "hard",
        root: str = ".",  # pylint: disable=unused-argument
        link: bool = True,
    ):
        """
        Clone object to `name`.

        :param name: new name
        :param how:
           - hard -> copy meta and artifacts to local fs
           - ref -> copy meta, reference remote artifacts
           - link -> make a link to remote meta
        :return: New meta file
        """
        if how != "hard":
            raise NotImplementedError()
        new: MlemMeta = deserialize(
            serialize(self, MlemMeta), MlemMeta
        )  # easier than deep copy bc of possible attached objects
        new.dump(name, mlem_root=new.name if link else None, link=link,
                 check_extension=False)  # only dump meta TODO: https://github.com/iterative/mlem/issues/37
        return new


class MlemLink(MlemMeta):
    mlem_link: str
    link_type: str
    object_type = "link"

    def load_link(self) -> T:
        linked_obj_path, linked_obj_fs = self.parse_link()
        return MlemMeta.__type_map__[self.link_type].read(
            linked_obj_path, fs=linked_obj_fs
        )

    def parse_link(self) -> Tuple[str, AbstractFileSystem]:
        if self.name is None:
            raise ValueError("Link is not saved")
        if not isinstance(self.mlem_link, str):
            raise NotImplementedError()
        fs, path = get_fs(self.mlem_link)
        if not isinstance(fs, LocalFileSystem) or (
            isinstance(fs, LocalFileSystem) and os.path.isabs(self.mlem_link)
        ):
            return path, fs
        fs = self.fs
        mlem_root = find_mlem_root(self.name, fs)
        path = os.path.join(mlem_root, self.mlem_link)
        return path, fs

    def dump(self, path: str, fs: Union[str, AbstractFileSystem] = None, mlem_root: Optional[str] = None,
             link: bool = True, check_extension: bool = True, absolute: bool = False):
        # TODO: use `fs` everywhere instead of `os`? https://github.com/iterative/mlem/issues/26
        fs, _ = resolve_fs(fs, path)
        if mlem_root:
            mlem_root = find_mlem_root(mlem_root, fs=fs)
        if link or mlem_root is None:
            if check_extension and not path.endswith(MLEM_EXT):
                raise ValueError(f"name={path} should ends with {MLEM_EXT}")
            path = path
        else:
            path = mlem_dir_path(
                path,
                fs=fs,
                obj_type=self.link_type,
                mlem_root=mlem_root,
            )
        # TODO: maybe this should be done on serialization step?
        # https://github.com/iterative/mlem/issues/48
        if not absolute:
            try:
                mlem_root_for_path = find_mlem_root(path)
            except MlemRootNotFound as exc:
                raise MlemError(
                    "You can't create relative links in folders outside of mlem root"
                ) from exc
            else:
                if os.path.abspath(mlem_root_for_path) == os.path.abspath(
                    find_mlem_root(self.mlem_link)
                ):
                    self.mlem_link = os.path.relpath(
                        self.mlem_link, start=mlem_root_for_path
                    )
        parent_dir = os.path.dirname(path)
        if parent_dir:
            fs.makedirs(parent_dir, exist_ok=True)
        with fs.open(path, "w") as f:
            safe_dump(serialize(self), f)
        self.name = path
        self.fs = fs


class _ExternalMeta(ABC, MlemMeta):
    artifacts: Optional[Artifacts] = None
    requirements: Requirements

    def dump(self, path: str, fs: Union[str, AbstractFileSystem] = None, mlem_root: Optional[str] = None,
             link: bool = True, check_extension: bool = True, absolute: bool = False):
        self.fs, _ = resolve_fs(fs, path)
        if mlem_root is not None:
            # check if name is relative here?
            path = os.path.join(mlem_root, path)
        if not path.endswith(MLEM_EXT):
            path = os.path.join(path, META_FILE_NAME)
        self.name = path
        self.artifacts = self.write_value(mlem_root or ".")
        super().dump(path=path, fs=fs, mlem_root=mlem_root, link=link, check_extension=check_extension,
                     absolute=absolute)

    @abstractmethod
    def write_value(self, mlem_root: str) -> Artifacts:
        raise NotImplementedError

    @property
    def art_dir(self):
        return os.path.join(os.path.dirname(self.name), ART_DIR)

    def ensure_saved(self):
        if self.fs is None:
            raise ValueError(f"Can't load {self}: it's not saved")

    def clone(
        self, name: str, how: str = "hard", root: str = ".", link: bool = True
    ):
        """

        :param name: new name
        :param how:
           - hard -> copy meta and artifacts to local fs
           - ref -> copy meta, reference remote artifacts
           - link -> make a link to remote meta
        :parameter root: where to store new model
        :param link: whether to make link to it in repo
        :return: New meta file
        """
        if how != "hard":
            # TODO: https://github.com/iterative/mlem/issues/37
            raise NotImplementedError()
        if self.name is None:
            raise ValueError("Cannot clone not saved object")
        mlem_root = find_mlem_root(root, raise_on_missing=False)
        new: _ExternalMeta = deserialize(
            serialize(self, MlemMeta), _ExternalMeta
        )  # easier than deep copy bc of possible attached objects
        if not name.endswith(MLEM_EXT):
            name = os.path.join(name, META_FILE_NAME)
        path = os.path.join(root, name)
        new.name = path

        os.makedirs(new.art_dir, exist_ok=True)
        new.artifacts = []
        for art in self.relative_artifacts:
            new.artifacts.append(art.download(new.art_dir))

        super(_ExternalMeta, new).dump(name, mlem_root=mlem_root if link else None, link=link and mlem_root is not None,
                                       check_extension=False)  # only dump meta TODO: https://github.com/iterative/mlem/issues/37
        return new

    @property
    def dirname(self):
        return os.path.dirname(self.name)

    @property
    def relative_artifacts(self) -> Artifacts:
        return [
            a.relative(self.fs, self.dirname) for a in self.artifacts or []
        ]

    @property
    def storage(self):
        if not self.fs or isinstance(self.fs, LocalFileSystem):
            return CONFIG.default_storage.relative(self.fs, self.dirname)
        return FSSpecStorage.from_fs_path(self.fs, self.dirname)


class ModelMeta(_ExternalMeta):
    object_type: ClassVar = "model"
    model_type_cache: Dict
    model_type: ModelType
    model_type, model_type_raw, model_type_cache = lazy_field(
        ModelType, "model_type", "model_type_cache"
    )

    @classmethod
    def from_obj(cls, model: Any, sample_data: Any = None) -> "ModelMeta":
        mt = ModelAnalyzer.analyze(model, sample_data=sample_data)
        mt.model = model
        return ModelMeta(
            model_type=mt, requirements=mt.get_requirements().expanded
        )

    def write_value(self, mlem_root: str) -> Artifacts:
        if self.model_type.model is not None:
            artifacts = self.model_type.io.dump(
                self.storage,
                ART_DIR,
                self.model_type.model,
            )
        else:
            raise NotImplementedError  # TODO: https://github.com/iterative/mlem/issues/37
            # self.get_artifacts().materialize(path)
        return artifacts

    def load_value(self):
        self.model_type.load(self.relative_artifacts)

    def get_value(self):
        return self.model_type.model

    def __getattr__(self, item):
        if item not in self.model_type.methods:
            raise AttributeError(
                f"{self.model_type.__class__} does not have {item} method"
            )
        return partial(self.model_type.call_method, item)


class DatasetMeta(_ExternalMeta):
    __transient_fields__ = {"dataset"}
    object_type: ClassVar = "dataset"
    reader_cache: Optional[Dict]
    reader: Optional[DatasetReader]
    reader, reader_raw, reader_cache = lazy_field(
        DatasetReader,
        "reader",
        "reader_cache",
        parse_as_type=Optional[DatasetReader],
        default=None,
    )
    dataset: ClassVar[Optional[Dataset]] = None

    @property
    def data(self):
        return self.dataset.data

    @classmethod
    def from_data(cls, data: Any) -> "DatasetMeta":
        dataset = Dataset.create(
            data,
        )
        meta = DatasetMeta(
            requirements=dataset.dataset_type.get_requirements().expanded
        )
        meta.dataset = dataset
        return meta

    def write_value(self, mlem_root: str) -> Artifacts:
        if self.dataset is not None:
            reader, artifacts = self.dataset.dataset_type.get_writer().write(
                self.dataset,
                self.storage,
                ART_DIR,
            )
            self.reader = reader
        else:
            raise NotImplementedError()  # TODO: https://github.com/iterative/mlem/issues/37
            # artifacts = self.get_artifacts()
        return artifacts

    def load_value(self):
        self.dataset = self.reader.read(self.relative_artifacts)
        # with self.localize(self.art_dir) as (fs, path):
        #     self.dataset = self.reader.read(fs, path)

    def get_value(self):
        return self.data

    # def get_artifacts(self) -> ArtifactCollection:
    #     return blobs_from_path(self.art_dir, self.fs)


class TargetEnvMeta(MlemMeta):
    __type_root__ = True
    object_type: ClassVar = "env"
    alias: ClassVar = ...
    deployment_type: ClassVar = ...

    additional_args = ()

    @abstractmethod
    def deploy(self, meta: "ModelMeta", **kwargs) -> "Deployment":
        """"""
        raise NotImplementedError()

    @abstractmethod
    def update(
        self, meta: "ModelMeta", previous: "Deployment", **kwargs
    ) -> "Deployment":
        """"""
        raise NotImplementedError()


@dataclass
class DeployMeta(MlemMeta):
    object_type: ClassVar = "deployment"

    env_path: str
    model_path: str
    deployment: Deployment

    @property  # TODO cached
    def env(self):
        return TargetEnvMeta.read(self.env_path, self.fs)

    @property  # TODO cached
    def model(self):
        return ModelMeta.read(self.model_path, self.fs)

    @classmethod
    def find(
        cls, env_path: str, model_path: str, raise_on_missing=True
    ) -> Optional["DeployMeta"]:
        try:
            path = os.path.join(env_path, model_path)
            return cls.read(path, LocalFileSystem())  # TODO fs
        except FileNotFoundError:
            if raise_on_missing:
                raise
            return None


def mlem_dir_path(
    name,
    fs: AbstractFileSystem,
    obj_type: Union[Type[MlemMeta], str],
    mlem_root: Optional[str] = None,
) -> str:
    """Construct path to object link in MLEM root dir

    Args:
        name ([type]): Path to the object.
        fs (AbstractFileSystem): FS where object is located.
        obj_type (Union[Type[MlemMeta], str]): Type of object.
        mlem_root (str, optional): Path to MLEM root dir. If not provided,
            we'll search mlem_root for given `name`.

    Returns:
        str: Path to the given object in MLEM root dir
    """
    if mlem_root is None:
        mlem_root = find_mlem_root(path=name, fs=fs)
    if not isinstance(obj_type, str):
        obj_type = obj_type.object_type
    if name.endswith(META_FILE_NAME) and not name.endswith(MLEM_EXT):
        name = os.path.dirname(name)
    if not name.endswith(MLEM_EXT):
        name += MLEM_EXT
    if os.path.abspath(mlem_root) in os.path.abspath(name):
        name = os.path.relpath(name, start=mlem_root)
    return os.path.join(mlem_root, MLEM_DIR, obj_type, name)


def find_object(
    path: str, fs: AbstractFileSystem, mlem_root: str = None
) -> Tuple[str, str]:
    """assumes .mlem/ content is valid"""
    if mlem_root is None:
        mlem_root = find_mlem_root(path, fs)
    source_paths = [
        (tp, mlem_dir_path(path, fs, obj_type=cls, mlem_root=mlem_root))
        for tp, cls in MlemMeta.__type_map__.items()
        if issubclass(cls, MlemMeta)
    ]
    source_paths = [p for p in source_paths if fs.exists(p[1])]
    if len(source_paths) == 0:
        raise ValueError(f"Object {path} not found, search of fs {fs}")
    if len(source_paths) > 1:
        raise ValueError(f"Ambiguous object {path}: {source_paths}")
    type, source_path = source_paths[0]
    return type, source_path
