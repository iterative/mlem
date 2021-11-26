"""
Base classes for meta objects in MLEM:
MlemMeta and it's subclasses, e.g. ModelMeta, DatasetMeta, etc
"""
import os
import posixpath
from abc import ABC, abstractmethod
from functools import partial
from inspect import isabstract
from typing import Any, ClassVar, Dict, Optional, Tuple, Type, TypeVar, Union

from fsspec import AbstractFileSystem
from fsspec.implementations.local import LocalFileSystem
from pydantic import BaseModel
from yaml import safe_dump, safe_load

from mlem.config import CONFIG
from mlem.core.artifacts import (
    Artifacts,
    FSSpecArtifact,
    FSSpecStorage,
    LocalArtifact,
)
from mlem.core.base import MlemObject
from mlem.core.dataset_type import Dataset, DatasetReader
from mlem.core.errors import MlemObjectNotSavedError, MlemRootNotFound
from mlem.core.meta_io import (
    ART_DIR,
    META_FILE_NAME,
    MLEM_DIR,
    MLEM_EXT,
    deserialize,
    get_fs,
    serialize,
)
from mlem.core.model import ModelAnalyzer, ModelType
from mlem.core.requirements import Requirements
from mlem.polydantic.lazy import lazy_field
from mlem.utils.path import make_posix
from mlem.utils.root import find_repo_root


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
    __transient_fields__ = {"_path", "_fs", "_repo"}
    __abstract__: ClassVar[bool] = True
    object_type: ClassVar[str]
    _path: Optional[str] = None
    _repo: Optional[str] = None
    _fs: ClassVar[Optional[AbstractFileSystem]] = None

    @property
    def name(self):
        """Name of the object in the repo"""
        repo_path = posixpath.relpath(self._path, self._repo)[: -len(MLEM_EXT)]
        prefix = posixpath.join(MLEM_DIR, self.object_type)
        if repo_path.startswith(prefix):
            repo_path = repo_path[len(prefix) + 1 :]
        return repo_path

    @property
    def is_saved(self):
        return self._path is not None

    @property
    def resolved_type(self):
        return self.object_type

    @classmethod
    def get_metafile_path(cls, fullpath: str):
        """Augment path to point to metafile, if it is not"""
        if not fullpath.endswith(MLEM_EXT):
            fullpath += MLEM_EXT
        return fullpath

    @classmethod
    def _get_location(
        cls,
        path: str,
        repo: Optional[str],
        fs: Optional[AbstractFileSystem],
        external: bool,
        ensure_mlem_root: bool,
        metafile_path: bool = True,
    ) -> Tuple[AbstractFileSystem, str, Optional[str]]:
        """Extract fs, path and mlem repo"""

        fullpath = posixpath.join(repo or "", path)
        if metafile_path:
            fullpath = cls.get_metafile_path(fullpath)
        if fs is None:
            fs, fullpath = get_fs(fullpath)
        elif isinstance(fs, LocalFileSystem):
            fullpath = make_posix(os.path.abspath(fullpath))
        if repo is not None:
            # check that repo is mlem repo root
            find_repo_root(repo, fs, raise_on_missing=True, recursive=False)
        else:
            repo = find_repo_root(fullpath, fs, raise_on_missing=False)
        if ensure_mlem_root and repo is None:
            raise MlemRootNotFound(fullpath, fs)
        if (
            repo is None
            or external
            or fullpath.startswith(
                posixpath.join(repo, MLEM_DIR, cls.object_type)
            )
        ):
            # orphan or external or inside .mlem
            return fs, fullpath, repo

        internal_path = posixpath.join(
            repo,
            MLEM_DIR,
            cls.object_type,
            os.path.relpath(fullpath, repo),
        )
        return fs, internal_path, repo

    def bind(
        self, path: str, fs: AbstractFileSystem, repo: Optional[str] = None
    ):
        self._path = path
        self._fs = fs
        self._repo = repo or find_repo_root(path, fs, False)
        return self

    @classmethod
    def read(
        cls: Type[T],
        path: str,
        fs: AbstractFileSystem = None,
        repo: str = None,
        follow_links: bool = True,
    ) -> T:
        """
        Read object in (path, fs)
            and try to deserialise it as `cls` instance

        Args:
            path: Exact path to MLEM metafile,
            fs: Filesystem on which path is located,
            repo: path to mlem repo, optional
            follow_links: If deserialised object is a MLEM link,
                whether to load and return the linked object
                or just return MlemLink object.

        Returns:
            Deserialised object
        """
        fs, path, repo = cls._get_location(path, repo, fs, True, False, False)
        with fs.open(path) as f:
            payload = safe_load(f)
        res = deserialize(payload, cls).bind(path, fs, repo)
        if follow_links and isinstance(res, MlemLink):
            link = res.load_link()
            if not isinstance(link, cls):
                raise ValueError(f"Wrong type inside link: {link.__class__}")
            return link
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
        repo: Optional[str] = None,
        link: Optional[bool] = None,
        external: Optional[bool] = None,
    ):
        """Dumps metafile and possible artifacts to path.

        Args:
            path: name of the object. Relative to repo, if it is provided.
            fs: filesystem to save to. if not provided, inferred from repo and path
            repo: path to mlem repo
            link: whether to create link if object is external.
                If set to True, checks existanse of mlem repo
                defaults to True if mlem repo exists and external is true
            external: whether to save object inside mlem dir or not.
                Defaults to false if repo is provided
                Forced to false if path points inside mlem dir
        """
        fullpath, repo, fs, link, external = self._parse_dump_args(
            path, repo, fs, link, external
        )
        self._write_meta(fullpath, repo, fs, link)

    def _write_meta(
        self,
        path: str,
        repo: Optional[str],
        fs: AbstractFileSystem,
        link: bool,
    ):
        """Write metadata to path in fs and possibly create link in mlem dir"""
        fs.makedirs(posixpath.dirname(path), exist_ok=True)
        with fs.open(path, "w") as f:
            safe_dump(serialize(self), f)
        if link and repo:
            self.make_link(self.name, fs, repo=repo, external=False)

    def _parse_dump_args(
        self,
        path: str,
        repo: Optional[str],
        fs: Optional[AbstractFileSystem],
        link: Optional[bool],
        external: Optional[bool],
    ) -> Tuple[str, Optional[str], AbstractFileSystem, bool, bool]:
        """Parse arguments for .dump and bind meta"""
        if external is None:
            external = CONFIG.DEFAULT_EXTERNAL
        # by default we make link only for external non-orphan objects
        if link is None:
            link = external
            ensure_mlem_root = False
        else:
            # if link manually set to True, there should be mlem repo
            ensure_mlem_root = link
        fs, fullpath, repo = self._get_location(
            make_posix(path), make_posix(repo), fs, external, ensure_mlem_root
        )
        self.bind(fullpath, fs, repo)
        if repo is not None:
            # force external=False if fullpath inside MLEM_DIR
            external = os.path.join(MLEM_DIR, "") not in os.path.dirname(
                fullpath
            )
        return fullpath, repo, fs, link and external, external

    def make_link(
        self,
        path: str = None,
        fs: Optional[AbstractFileSystem] = None,
        repo: Optional[str] = None,
        external: Optional[bool] = None,
        absolute: bool = False,
    ) -> "MlemLink":
        if not self.is_saved:
            raise MlemObjectNotSavedError(
                "Cannot create link for not saved meta object"
            )

        link = MlemLink(
            link_data=LinkData(
                path=self._path
                if absolute
                else self.get_metafile_path(self.name)
            ),
            link_type=self.resolved_type,
        )
        if path is not None:
            link.dump(path, fs, repo, external=external, link=False)
        return link

    @classmethod
    def non_abstract_subtypes(cls) -> Dict[str, Type["MlemMeta"]]:
        return {
            k: v
            for k, v in cls.__type_map__.items()
            if not isabstract(v) and not v.__dict__.get("__abstract__", False)
        }

    def clone(
        self,
        path: str,
        fs: Union[str, AbstractFileSystem, None] = None,
        repo: Optional[str] = None,
        link: Optional[bool] = None,
        external: Optional[bool] = None,
    ):
        """
        Clone existing object to `path`.

        Arguments are the same as for `dump`
        """
        if not self.is_saved:
            raise MlemObjectNotSavedError("Cannot clone not saved object")
        new: "MlemMeta" = self.deepcopy()
        new.dump(
            path, fs, repo, link, external
        )  # only dump meta TODO: https://github.com/iterative/mlem/issues/37
        return new

    def deepcopy(self):
        return deserialize(
            serialize(self, MlemMeta), MlemMeta
        )  # easier than deep copy bc of possible attached objects

    # def update(self):
    #     if not self.is_saved:
    #         raise MlemObjectNotSavedError("Cannot update not saved object")
    #     self._write_meta(self._path, self._root, self._fs, False)


class LinkData(BaseModel):
    path: str
    repo: Optional[str] = None
    rev: Optional[str] = None


class MlemLink(MlemMeta):
    link_data: LinkData
    link_type: str
    object_type = "link"

    @property
    def link_cls(self) -> Type[MlemMeta]:
        return MlemMeta.__type_map__[self.link_type]

    @property
    def resolved_type(self):
        return self.link_type

    def load_link(self) -> MlemMeta:
        linked_obj_path, linked_obj_fs = self.parse_link()
        obj = self.link_cls.read(linked_obj_path, fs=linked_obj_fs)
        return obj

    def parse_link(self) -> Tuple[str, AbstractFileSystem]:
        if not self.is_saved:
            raise MlemObjectNotSavedError("Link is not saved")
        fs, path = get_fs(self.link_data.path)
        if not isinstance(fs, LocalFileSystem) or (
            isinstance(fs, LocalFileSystem)
            and os.path.isabs(self.link_data.path)
        ):
            return path, fs
        from mlem.core.metadata import find_meta_path

        return (
            find_meta_path(
                posixpath.join(self._repo or "", self.link_data.path), self._fs
            ),
            self._fs,
        )


class _WithArtifacts(ABC, MlemMeta):
    __abstract__: ClassVar[bool] = True
    artifacts: Optional[Artifacts] = None
    requirements: Requirements

    @classmethod
    def get_metafile_path(cls, fullpath: str):
        """Augment fullpath to point to metafile, if it is not"""
        if not fullpath.endswith(META_FILE_NAME):
            fullpath = posixpath.join(fullpath, META_FILE_NAME)
        return fullpath

    @property
    def name(self):
        repo_path = os.path.dirname(os.path.relpath(self._path, self._repo))
        prefix = os.path.join(MLEM_DIR, self.object_type)
        if repo_path.startswith(prefix):
            repo_path = repo_path[len(prefix) + 1 :]
        return repo_path

    def dump(
        self,
        path: str,
        fs: Union[str, AbstractFileSystem, None] = None,
        repo: Optional[str] = None,
        link: Optional[bool] = None,
        external: Optional[bool] = None,
    ):
        fullpath, repo, fs, link, external = self._parse_dump_args(
            path, repo, fs, link, external
        )
        self.artifacts = self.write_value()
        self._write_meta(fullpath, repo, fs, link)

    @abstractmethod
    def write_value(self) -> Artifacts:
        raise NotImplementedError

    @property
    def art_dir(self):
        return posixpath.join(os.path.dirname(self._path), ART_DIR)

    # def ensure_saved(self):
    #     if self.fs is None:
    #         raise ValueError(f"Can't load {self}: it's not saved")

    def clone(
        self,
        path: str,
        fs: Union[str, AbstractFileSystem, None] = None,
        repo: Optional[str] = None,
        link: Optional[bool] = None,
        external: Optional[bool] = None,
    ):
        if not self.is_saved:
            raise MlemObjectNotSavedError("Cannot clone not saved object")
        # clone is just dump with copying artifacts
        new: "_WithArtifacts" = self.deepcopy()
        new.artifacts = []
        (
            fullpath,
            repo,
            fs,
            link,
            external,
        ) = new._parse_dump_args(  # pylint: disable=protected-access
            path, repo, fs, link, external
        )
        fs.makedirs(new.art_dir, exist_ok=True)
        for art in self.relative_artifacts:
            download = art.materialize(
                new.art_dir, new._fs  # pylint: disable=protected-access
            )
            if isinstance(download, FSSpecArtifact):
                download = LocalArtifact(
                    uri=posixpath.relpath(download.uri, make_posix(path))
                )
            new.artifacts.append(download)
        new._write_meta(  # pylint: disable=protected-access
            fullpath, repo, fs, link
        )
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

    def __getattr__(self, item):
        if item not in self.model_type.methods:
            raise AttributeError(
                f"{self.model_type.__class__} does not have {item} method"
            )
        return partial(self.model_type.call_method, item)


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
            path = posixpath.join(env_path, model_path)
            return cls.read(path, LocalFileSystem())  # TODO fs
        except FileNotFoundError:
            if raise_on_missing:
                raise
            return None


def mlem_dir_path(
    name: str,
    fs: Optional[AbstractFileSystem],
    obj_type: Union[Type[MlemMeta], str],
    repo: Optional[str] = None,
) -> str:
    """Construct path to object link in MLEM root dir

    Args:
        name ([type]): Path to the object.
        fs (AbstractFileSystem): FS where object is located.
        obj_type (Union[Type[MlemMeta], str]): Type of object.
        repo (str, optional): Path to MLEM root dir. If not provided,
            we'll search mlem_root for given `name`.

    Returns:
        str: Path to the given object in MLEM root dir
    """
    if repo is None:
        repo = find_repo_root(path=name, fs=fs)
    if not isinstance(obj_type, str):
        obj_type = obj_type.object_type
    if name.endswith(META_FILE_NAME) and not name.endswith(MLEM_EXT):
        name = os.path.dirname(name)
    if not name.endswith(MLEM_EXT):
        name += MLEM_EXT
    if os.path.abspath(repo) in os.path.abspath(name):
        name = os.path.relpath(name, start=repo)
    return posixpath.join(repo, MLEM_DIR, obj_type, name)


def find_object(
    path: str, fs: AbstractFileSystem, repo: str = None
) -> Tuple[str, str]:
    """assumes .mlem/ content is valid"""
    if repo is None:
        repo = find_repo_root(path, fs)
    if repo is not None and path.startswith(repo):
        path = os.path.relpath(path, repo)
    source_paths = [
        (
            tp,
            posixpath.join(
                repo or "",
                MLEM_DIR,
                cls.object_type,
                cls.get_metafile_path(path),
            ),
        )
        for tp, cls in MlemMeta.non_abstract_subtypes().items()
    ]
    source_paths = [p for p in set(source_paths) if fs.exists(p[1])]
    if len(source_paths) == 0:
        raise ValueError(
            f"Object {path} not found, search of fs {fs} at {path}"
        )
    if len(source_paths) > 1:
        raise ValueError(f"Ambiguous object {path}: {source_paths}")
    type, source_path = source_paths[0]
    return type, source_path
