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
    __transient_fields__ = {"_path", "_fs", "_root"}
    object_type: ClassVar[str]
    _path: Optional[str] = None
    _root: Optional[str] = None
    _fs: ClassVar[Optional[AbstractFileSystem]] = None

    @property
    def name(self):
        """Name of the object in the repo"""
        return os.path.relpath(self._path, self._root)[: -len(MLEM_EXT)]

    @property
    def is_saved(self):
        return self._path is not None

    @classmethod
    def _make_meta_path(cls, fullpath: str):
        """Augment path to point to metafile, if it is not"""
        if not fullpath.endswith(MLEM_EXT):
            fullpath += MLEM_EXT
        return fullpath

    @classmethod
    def _get_location(
        cls,
        path: str,
        mlem_root: str,
        fs: AbstractFileSystem,
        external: bool,
        ensure_mlem_root: bool,
        create_meta_path: bool = True
    ):
        """Extract fs, path and mlem_root"""
        fullpath = os.path.join(mlem_root or "", path)
        if create_meta_path:
            fullpath = cls._make_meta_path(fullpath)
        if fs is None:
            fs, fullpath = get_fs(fullpath)
        elif isinstance(fs, LocalFileSystem):
            fullpath = os.path.abspath(fullpath)

        # here we allow for mlem_root to be anywhere upper than fullpath
        # but maybe we need to be more strict and check if mlem_root is actually has .mlem dir
        mlem_root_ = find_mlem_root(fullpath, fs, raise_on_missing=False)
        if ensure_mlem_root and mlem_root_ is None:
            raise MlemRootNotFound(fullpath, fs)
        if mlem_root is None and mlem_root_ is not None:
            # we were fiven fullpath from the beginning
            external = True
        if mlem_root_ is None or external:
            # orphan or external
            return fs, fullpath, mlem_root_

        internal_path = os.path.join(
            mlem_root_,
            MLEM_DIR,
            cls.object_type,
            os.path.relpath(fullpath, mlem_root_),
        )
        return fs, internal_path, mlem_root_

    def bind(
        self, path: str, fs: AbstractFileSystem, root: Optional[str] = None
    ):
        self._path = path
        self._fs = fs
        self._root = root or find_mlem_root(path, fs, False)
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
        fs, path, root = cls._get_location(path, mlem_root, fs, True, False, False)
        with fs.open(path) as f:
            payload = safe_load(f)
        res = deserialize(payload, cls).bind(path, fs, root)
        if follow_links and isinstance(res, MlemLink):
            return res.load_link()
        return res

    def write_value(self) -> Optional[Artifacts]:
        return None

    def load_value(self):
        pass

    def unload_value(self):
        pass

    def get_value(self):
        return self

    def dump(
        self,
        path: str,
        fs: Union[str, AbstractFileSystem] = None,
        mlem_root: Optional[str] = None,
        link: bool = ...,
        external: bool = ...,
    ):
        """Dumps metafile and possible artifacts to path.

        Args:
            path: name of the object. Relative to mlem_root, if it is provided.

            fs
            mlem_root
            link
            external
        """
        fullpath, mlem_root, fs, link, external = self._parse_dump_args(
            path, mlem_root, fs, link, external
        )
        external = external and not fullpath.startswith(
            os.path.join(mlem_root, MLEM_DIR)
        )
        self._write_meta(fullpath, mlem_root, fs, link and external)

    def _write_meta(self, path:str, mlem_root:Optional[str], fs:AbstractFileSystem, link:bool):
        """Write metadata to path in fs and possibly create link in mlem_root"""
        fs.makedirs(os.path.dirname(path), exist_ok=True)
        with fs.open(path, "w") as f:
            safe_dump(serialize(self), f)
        if link and mlem_root:
            self.make_link(
                os.path.join(mlem_root, MLEM_DIR, self.object_type, self.name),
                fs,
                mlem_root=mlem_root,
            )

    def _parse_dump_args(self, path, mlem_root, fs, link, external):
        """Parse arguments for .dump and bind meta"""
        if external is ...:
            # TODO default from settings
            external = False
        # by default we make link only for external non-orphan objects
        if link is ...:
            link = external
            ensure_mlem_root = False
        else:
            # if link manually set to True, there should be mlem repo
            ensure_mlem_root = link
        fs, fullpath, mlem_root = self._get_location(
            path, mlem_root, fs, external, ensure_mlem_root
        )
        self.bind(fullpath, fs, mlem_root)
        return fullpath, mlem_root, fs, link, external

    def make_link(
        self,
        path: str = None,
        fs: AbstractFileSystem = None,
        mlem_root: str = None,
    ) -> "MlemLink":
        if not self.is_saved:
            raise MlemObjectNotSavedError(
                "Cannot create link for not saved meta object"
            )

        link = MlemLink(mlem_link=self._path, link_type=self.object_type)
        link.dump(path, fs, mlem_root)
        return link

    @classmethod
    def subtype_mapping(cls) -> Dict[str, Type["MlemMeta"]]:
        return cls.__type_map__

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
        raise NotImplementedError  # TODO
        if how != "hard":
            raise NotImplementedError()
        new: MlemMeta = deserialize(
            serialize(self, MlemMeta), MlemMeta
        )  # easier than deep copy bc of possible attached objects
        new.dump(
            name,
            mlem_root=new.name if link else None,
            link=link,
            check_extension=False,
        )  # only dump meta TODO: https://github.com/iterative/mlem/issues/37
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
        if not self.is_saved:
            raise MlemObjectNotSavedError("Link is not saved")
        if not isinstance(self.mlem_link, str):
            raise NotImplementedError
        fs, path = get_fs(self.mlem_link)
        if not isinstance(fs, LocalFileSystem) or (
            isinstance(fs, LocalFileSystem) and os.path.isabs(self.mlem_link)
        ):
            return path, fs
        fs = self.fs
        mlem_root = find_mlem_root(self.name, fs)
        path = os.path.join(mlem_root, self.mlem_link)
        return path, fs

    def dump(
        self,
        path: str,
        fs: Union[str, AbstractFileSystem] = None,
        mlem_root: Optional[str] = None,
        link: bool = ...,
        external: bool = ...,
    ):
        # for now, links should always be saved via fullpath (as if external=True), even if they are in mlem dir
        # links also do not create links to them in index
        # we can change it in the future
        fs, fullpath, mlem_root = self._get_location(
            path, mlem_root, fs, True, False
        )
        self.bind(fullpath, fs, mlem_root)
        self._write_meta(fullpath, mlem_root, fs, link=False)

    # TODO remove?
    # def dump(
    #     self,
    #     path: str,
    #     fs: Union[str, AbstractFileSystem] = None,
    #     mlem_root: Optional[str] = None,
    #     link: bool = True,
    #     check_extension: bool = True,
    #     absolute: bool = False,
    # ):
    #     # TODO: use `fs` everywhere instead of `os`? https://github.com/iterative/mlem/issues/26
    #     fs, _ = resolve_fs(fs, path)
    #     if mlem_root:
    #         mlem_root = find_mlem_root(mlem_root, fs=fs)
    #     if link or mlem_root is None:
    #         if check_extension and not path.endswith(MLEM_EXT):
    #             raise ValueError(f"name={path} should ends with {MLEM_EXT}")
    #         path = path
    #     else:
    #         path = mlem_dir_path(
    #             path,
    #             fs=fs,
    #             obj_type=self.link_type,
    #             mlem_root=mlem_root,
    #         )
    #     # TODO: maybe this should be done on serialization step?
    #     # https://github.com/iterative/mlem/issues/48
    #     if not absolute:
    #         try:
    #             mlem_root_for_path = find_mlem_root(path)
    #         except MlemRootNotFound as exc:
    #             raise MlemError(
    #                 "You can't create relative links in folders outside of mlem root"
    #             ) from exc
    #         else:
    #             if os.path.abspath(mlem_root_for_path) == os.path.abspath(
    #                 find_mlem_root(self.mlem_link)
    #             ):
    #                 self.mlem_link = os.path.relpath(
    #                     self.mlem_link, start=mlem_root_for_path
    #                 )
    #     parent_dir = os.path.dirname(path)
    #     if parent_dir:
    #         fs.makedirs(parent_dir, exist_ok=True)
    #     with fs.open(path, "w") as f:
    #         safe_dump(serialize(self), f)
    #     self.name = path
    #     self.fs = fs


class _WithArtifacts(ABC, MlemMeta):
    artifacts: Optional[Artifacts] = None
    requirements: Requirements

    @classmethod
    def _make_meta_path(cls, fullpath: str):
        """Augment fullpath to point to metafile, if it is not"""
        if not fullpath.endswith(META_FILE_NAME):
            fullpath = os.path.join(fullpath, META_FILE_NAME)
        return fullpath

    @property
    def name(self):
        return os.path.dirname(os.path.relpath(self._path, self._root))

    def dump(
        self,
        path: str,
        fs: Union[str, AbstractFileSystem] = None,
        mlem_root: Optional[str] = None,
        link: bool = ...,
        external: bool = ...,
    ):
        fullpath, mlem_root, fs, link, external = self._parse_dump_args(
            path, mlem_root, fs, link, external
        )
        self.artifacts = self.write_value()
        external = external and not fullpath.startswith(
            os.path.join(mlem_root, MLEM_DIR)
        )
        self._write_meta(fullpath, mlem_root, fs, link and external)

    @abstractmethod
    def write_value(self) -> Artifacts:
        raise NotImplementedError

    @property
    def art_dir(self):
        return os.path.join(os.path.dirname(self.name), ART_DIR)

    # def ensure_saved(self):
    #     if self.fs is None:
    #         raise ValueError(f"Can't load {self}: it's not saved")

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
        raise NotImplementedError  # TODO
        if how != "hard":
            # TODO: https://github.com/iterative/mlem/issues/37
            raise NotImplementedError()
        if self.name is None:
            raise ValueError("Cannot clone not saved object")
        mlem_root = find_mlem_root(root, raise_on_missing=False)
        new: _WithArtifacts = deserialize(
            serialize(self, MlemMeta), _WithArtifacts
        )  # easier than deep copy bc of possible attached objects
        if not name.endswith(MLEM_EXT):
            name = os.path.join(name, META_FILE_NAME)
        path = os.path.join(root, name)
        new.name = path

        os.makedirs(new.art_dir, exist_ok=True)
        new.artifacts = []
        for art in self.relative_artifacts:
            new.artifacts.append(art.download(new.art_dir))

        super(_WithArtifacts, new).dump(
            name,
            mlem_root=mlem_root if link else None,
            link=link and mlem_root is not None,
            check_extension=False,
        )  # only dump meta TODO: https://github.com/iterative/mlem/issues/37
        return new

    @property
    def dirname(self):
        return os.path.dirname(self._path)

    @property
    def relative_artifacts(self) -> Artifacts:
        return [
            a.relative(self._fs, self.dirname) for a in self.artifacts or []
        ]

    @property
    def storage(self):
        if not self._fs or isinstance(self._fs, LocalFileSystem):
            return CONFIG.default_storage.relative(self._fs, self.dirname)
        return FSSpecStorage.from_fs_path(self._fs, self.dirname)


class ModelMeta(_WithArtifacts):
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

    def write_value(self) -> Artifacts:
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

    # def __getattr__(self, item):
    #     if item not in self.model_type.methods:
    #         raise AttributeError(
    #             f"{self.model_type.__class__} does not have {item} method"
    #         )
    #     return partial(self.model_type.call_method, item)


class DatasetMeta(_WithArtifacts):
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

    def write_value(self) -> Artifacts:
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

    def get_value(self):
        return self.data


class TargetEnvMeta(MlemMeta):
    __type_root__ = True
    object_type: ClassVar = "env"
    alias: ClassVar = ...
    deployment_type: ClassVar = ...

    additional_args = ()

    @abstractmethod
    def deploy(self, meta: "ModelMeta", **kwargs) -> "Deployment":
        """"""
        raise NotImplementedError

    @abstractmethod
    def update(
        self, meta: "ModelMeta", previous: "Deployment", **kwargs
    ) -> "Deployment":
        """"""
        raise NotImplementedError


@dataclass
class DeployMeta(MlemMeta):
    object_type: ClassVar = "deployment"

    env_path: str
    model_path: str
    deployment: Deployment

    @property  # TODO cached
    def env(self):
        return TargetEnvMeta.read(self.env_path, self._fs)

    @property  # TODO cached
    def model(self):
        return ModelMeta.read(self.model_path, self._fs)

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
