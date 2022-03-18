"""
Base classes for meta objects in MLEM:
MlemMeta and it's subclasses, e.g. ModelMeta, DatasetMeta, etc
"""
import os
import posixpath
import time
from abc import ABC, abstractmethod
from enum import Enum
from functools import partial
from inspect import isabstract
from typing import (
    Any,
    ClassVar,
    Dict,
    Generic,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)

from fsspec import AbstractFileSystem
from fsspec.implementations.local import LocalFileSystem
from pydantic import parse_obj_as, validator
from typing_extensions import Literal
from yaml import safe_dump, safe_load

from mlem.config import CONFIG
from mlem.core.artifacts import (
    Artifacts,
    FSSpecArtifact,
    FSSpecStorage,
    LocalArtifact,
    PlaceholderArtifact,
)
from mlem.core.base import MlemObject
from mlem.core.dataset_type import Dataset, DatasetReader
from mlem.core.errors import (
    DeploymentError,
    MlemObjectNotFound,
    MlemObjectNotSavedError,
    MlemRootNotFound,
    WrongMetaType,
)
from mlem.core.meta_io import (
    MLEM_DIR,
    MLEM_EXT,
    Location,
    UriResolver,
    get_path_by_fs_path,
)
from mlem.core.model import ModelAnalyzer, ModelType
from mlem.core.requirements import Requirements
from mlem.polydantic.lazy import lazy_field
from mlem.utils.path import make_posix
from mlem.utils.root import find_repo_root

T = TypeVar("T", bound="MlemMeta")


class MlemMeta(MlemObject):
    class Config:
        exclude = {"location"}
        type_root = True
        type_field = "object_type"

    abs_name: ClassVar[str] = "meta"
    __abstract__: ClassVar[bool] = True
    object_type: ClassVar[str]
    location: Optional[Location] = None
    description: Optional[str] = None
    params: Dict[str, str] = {}
    tags: List[str] = []

    @property
    def loc(self) -> Location:
        if self.location is None:
            raise MlemObjectNotSavedError("Not saved object has no location")
        return self.location

    @property
    def name(self):
        """Name of the object in the repo"""
        repo_path = self.loc.path_in_repo[: -len(MLEM_EXT)]
        prefix = posixpath.join(MLEM_DIR, self.object_type)
        if repo_path.startswith(prefix):
            repo_path = repo_path[len(prefix) + 1 :]
        return repo_path

    @property
    def is_saved(self):
        return self.location is not None

    @property
    def resolved_type(self):
        return self.object_type

    @classmethod
    def get_metafile_path(cls, fullpath: str):
        """Augment path to point to metafile, if it is not"""
        if not fullpath.endswith(MLEM_EXT):
            fullpath += MLEM_EXT
        return fullpath

    def bind(self, location: Location):
        self.location = location
        return self

    @classmethod
    def _get_location(
        cls,
        path: str,
        repo: Optional[str],
        fs: Optional[AbstractFileSystem],
        external: bool,
        ensure_mlem_root: bool,
        metafile_path: bool = True,
    ) -> Location:
        """Create location from arguments"""
        if metafile_path:
            path = cls.get_metafile_path(path)
        loc = UriResolver.resolve(path, repo, rev=None, fs=fs, find_repo=True)
        if loc.repo is not None:
            # check that repo is mlem repo root
            find_repo_root(
                loc.repo, loc.fs, raise_on_missing=True, recursive=False
            )
        if ensure_mlem_root and loc.repo is None:
            raise MlemRootNotFound(loc.fullpath, loc.fs)
        if (
            loc.repo is None
            or external
            or loc.fullpath.startswith(
                posixpath.join(loc.repo, MLEM_DIR, cls.object_type)
            )
        ):
            # orphan or external or inside .mlem
            return loc

        internal_path = posixpath.join(
            MLEM_DIR,
            cls.object_type,
            loc.path_in_repo,
        )
        loc.update_path(internal_path)
        return loc

    @classmethod
    def read(
        cls: Type[T],
        location: Location,
        follow_links: bool = True,
    ) -> T:
        """
        Read object in (path, fs)
            and try to deserialise it as `cls` instance

        Args:
            location: location of metafile
            follow_links: If deserialised object is a MLEM link,
                whether to load and return the linked object
                or just return MlemLink object.

        Returns:
            Deserialised object
        """
        with location.open() as f:
            payload = safe_load(f)
        res = parse_obj_as(MlemMeta, payload).bind(location)
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
        location, link = self._parse_dump_args(path, repo, fs, link, external)
        self._write_meta(location, link)
        return self

    def _write_meta(
        self,
        location: Location,
        link: bool,
    ):
        """Write metadata to path in fs and possibly create link in mlem dir"""
        location.fs.makedirs(
            posixpath.dirname(location.fullpath), exist_ok=True
        )
        with location.open("w") as f:
            safe_dump(self.dict(), f)
        if link and location.repo:
            self.make_link(
                self.name, location.fs, repo=location.repo, external=False
            )

    def _parse_dump_args(
        self,
        path: str,
        repo: Optional[str],
        fs: Optional[AbstractFileSystem],
        link: Optional[bool],
        external: Optional[bool],
    ) -> Tuple[Location, bool]:
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
        location = self._get_location(
            make_posix(path), make_posix(repo), fs, external, ensure_mlem_root
        )
        self.bind(location)
        if location.repo is not None:
            # force external=False if fullpath inside MLEM_DIR
            external = posixpath.join(MLEM_DIR, "") not in posixpath.dirname(
                location.fullpath
            )
        return location, link and external

    def make_link(
        self,
        path: str = None,
        fs: Optional[AbstractFileSystem] = None,
        repo: Optional[str] = None,
        external: Optional[bool] = None,
        absolute: bool = False,
    ) -> "MlemLink":
        if self.location is None:
            raise MlemObjectNotSavedError(
                "Cannot create link for not saved meta object"
            )
        if absolute:
            link = MlemLink(
                path=self.loc.path,
                repo=self.loc.repo_uri,
                rev=self.loc.rev,
                link_type=self.resolved_type,
            )
        else:
            link = MlemLink(
                path=self.get_metafile_path(self.name),
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
            if not isabstract(v)
            and not v.__dict__.get("__abstract__", False)
            or v.__is_root__
            and v is not MlemMeta
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
        return parse_obj_as(
            MlemMeta, self.dict()
        )  # easier than deep copy bc of possible attached objects

    def update(self):
        if not self.is_saved:
            raise MlemObjectNotSavedError("Cannot update not saved object")
        self._write_meta(self.location, False)


class MlemLink(MlemMeta):
    path: str
    repo: Optional[str] = None
    rev: Optional[str] = None
    link_type: str

    object_type: ClassVar = "link"

    @property
    def link_cls(self) -> Type[MlemMeta]:
        return MlemMeta.__type_map__[self.link_type]

    @property
    def resolved_type(self):
        return self.link_type

    @validator("path", "repo")
    def make_posix(  # pylint: disable=no-self-argument
        cls, value  # noqa: B902
    ):
        return make_posix(value)

    @overload
    def load_link(
        self, follow_links: bool = True, *, force_type: Type[T]
    ) -> T:
        ...

    @overload
    def load_link(
        self, follow_links: bool = True, *, force_type: Literal[None] = None
    ) -> MlemMeta:
        ...

    def load_link(
        self, follow_links: bool = True, *, force_type: Type[MlemMeta] = None
    ) -> MlemMeta:
        if force_type is not None and self.link_cls != force_type:
            raise WrongMetaType(self.link_type, force_type)
        return self.link_cls.read(self.parse_link(), follow_links=follow_links)

    def parse_link(self) -> Location:
        from mlem.core.metadata import find_meta_location

        if self.repo is None and self.rev is None:
            # is it possible to have rev without repo?
            location = UriResolver.resolve(
                path=self.path, repo=None, rev=None, fs=None
            )
            if (
                location.repo is None
                and isinstance(location.fs, LocalFileSystem)
                and not os.path.isabs(
                    self.path
                )  # os is used for absolute win paths like c:/...
            ):
                # link is relative
                if self.location is None:
                    raise MlemObjectNotSavedError("Relative link is not saved")
                location = self.location.copy()
                location.update_path(self.path)
            return find_meta_location(location)
        # link is absolute
        return find_meta_location(
            UriResolver.resolve(
                path=self.path, repo=self.repo, rev=self.rev, fs=None
            )
        )

    @classmethod
    def from_location(
        cls, loc: Location, link_type: Union[str, Type[MlemMeta]]
    ) -> "MlemLink":
        return MlemLink(
            path=get_path_by_fs_path(loc.fs, loc.path_in_repo),
            repo=loc.repo,
            rev=loc.rev,
            link_type=link_type.object_type
            if not isinstance(link_type, str)
            else link_type,
        )


class _WithArtifacts(ABC, MlemMeta):
    __abstract__: ClassVar[bool] = True
    artifacts: Optional[Artifacts] = None
    requirements: Requirements = Requirements.new()

    @classmethod
    def get_metafile_path(cls, fullpath: str):
        """Augment fullpath to point to metafile, if it is not"""
        if not fullpath.endswith(MLEM_EXT):
            fullpath += MLEM_EXT
        return fullpath

    @property
    def name(self):
        repo_path = self.location.path_in_repo
        prefix = posixpath.join(MLEM_DIR, self.object_type)
        if repo_path.startswith(prefix):
            repo_path = repo_path[len(prefix) + 1 :]
        if repo_path.endswith(MLEM_EXT):
            repo_path = repo_path[: -len(MLEM_EXT)]
        return repo_path

    def dump(
        self,
        path: str,
        fs: Union[str, AbstractFileSystem, None] = None,
        repo: Optional[str] = None,
        link: Optional[bool] = None,
        external: Optional[bool] = None,
    ):
        location, link = self._parse_dump_args(path, repo, fs, link, external)
        try:
            existing = MlemMeta.read(location, follow_links=False)
            if isinstance(existing, _WithArtifacts):
                for art in existing.relative_artifacts:
                    art.remove()
        except (MlemObjectNotFound, FileNotFoundError):
            pass
        self.artifacts = self.get_artifacts()
        self._write_meta(location, link)
        return self

    @abstractmethod
    def write_value(self) -> Artifacts:
        raise NotImplementedError

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
        if self.location is None:
            raise MlemObjectNotSavedError("Cannot clone not saved object")
        # clone is just dump with copying artifacts
        new: "_WithArtifacts" = self.deepcopy()
        new.artifacts = []
        (
            location,
            link,
        ) = new._parse_dump_args(  # pylint: disable=protected-access
            path, repo, fs, link, external
        )
        for art in self.relative_artifacts:
            download = art.materialize(
                new.name, new.loc.fs  # pylint: disable=protected-access
            )
            if isinstance(download, FSSpecArtifact):
                download = LocalArtifact(
                    uri=posixpath.relpath(
                        download.uri, posixpath.dirname(make_posix(path))
                    ),
                    size=download.size,
                    hash=download.hash,
                )
            new.artifacts.append(download)
        new._write_meta(location, link)  # pylint: disable=protected-access
        return new

    @property
    def dirname(self):
        return os.path.dirname(self.location.fullpath)

    @property
    def relative_artifacts(self) -> Artifacts:
        if self.location is None:
            raise MlemObjectNotSavedError(
                "Cannot get relative artifacts for not saved object"
            )
        return [
            a.relative(self.location.fs, self.dirname)
            for a in self.artifacts or []
        ]

    @property
    def storage(self):
        if not self.location.fs or isinstance(
            self.location.fs, LocalFileSystem
        ):
            return CONFIG.default_storage.relative(
                self.location.fs, self.dirname
            )
        return FSSpecStorage.from_fs_path(self.location.fs, self.dirname)

    def get_artifacts(self):
        if self.artifacts is None:
            return self.write_value()
        return [
            a.relative_to(self.loc)
            if isinstance(a, PlaceholderArtifact)
            else a
            for a in self.artifacts
        ]


class ModelMeta(_WithArtifacts):
    object_type: ClassVar = "model"
    model_type_cache: Any
    model_type: ModelType
    model_type, model_type_raw, model_type_cache = lazy_field(
        ModelType, "model_type", "model_type_cache"
    )

    @classmethod
    def from_obj(
        cls,
        model: Any,
        sample_data: Any = None,
        description: str = None,
        tags: List[str] = None,
        params: Dict[str, str] = None,
    ) -> "ModelMeta":
        mt = ModelAnalyzer.analyze(model, sample_data=sample_data)
        mt.model = model
        return ModelMeta(
            model_type=mt,
            requirements=mt.get_requirements().expanded,
            description=description,
            tags=tags or [],
            params=params or {},
        )

    def write_value(self) -> Artifacts:
        if self.model_type.model is not None:
            return self.model_type.io.dump(
                self.storage,
                posixpath.basename(self.name),
                self.model_type.model,
            )
        raise ValueError("Meta is not binded to actual model")

    def load_value(self):
        with self.requirements.import_custom():
            self.model_type.load(self.relative_artifacts)

    def get_value(self):
        return self.model_type.model

    def __getattr__(self, item):
        if item not in self.model_type.methods:
            raise AttributeError(
                f"{self.model_type.__class__} does not have {item} attribute"
            )
        return partial(self.model_type.call_method, item)


class DatasetMeta(_WithArtifacts):
    class Config:
        exclude = {"dataset"}

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
    dataset: Optional[Dataset] = None

    @property
    def data(self):
        return self.dataset.data

    @classmethod
    def from_data(
        cls,
        data: Any,
        description: str = None,
        params: Dict[str, str] = None,
        tags: List[str] = None,
    ) -> "DatasetMeta":
        dataset = Dataset.create(
            data,
        )
        meta = DatasetMeta(
            requirements=dataset.dataset_type.get_requirements().expanded,
            description=description,
            params=params or {},
            tags=tags or [],
        )
        meta.dataset = dataset
        return meta

    def write_value(self) -> Artifacts:
        if self.dataset is not None:
            reader, artifacts = self.dataset.dataset_type.get_writer().write(
                self.dataset,
                self.storage,
                os.path.basename(self.name),
            )
            self.reader = reader
            return artifacts
        raise ValueError("Meta is not binded to actual data")

    def load_value(self):
        self.dataset = self.reader.read(self.relative_artifacts)

    def get_value(self):
        return self.data


class DeployState(MlemObject):
    class Config:
        type_root = True

    abs_name: ClassVar[str] = "deploy_state"

    # @abstractmethod
    # def get_client(self) -> BaseClient:
    #     pass


DT = TypeVar("DT", bound="DeployMeta")


class TargetEnvMeta(MlemMeta, Generic[DT]):
    class Config:
        type_root = True
        type_field = "type"

    abs_name = "env"
    object_type: ClassVar = "env"
    type: ClassVar = ...
    deploy_type: ClassVar[Type[DT]]

    @abstractmethod
    def deploy(self, meta: DT):
        """"""
        raise NotImplementedError

    @abstractmethod
    def destroy(self, meta: DT):
        """"""
        raise NotImplementedError

    @abstractmethod
    def get_status(self, meta: DT, raise_on_error=True) -> "DeployStatus":
        raise NotImplementedError

    def check_type(self, deploy: "DeployMeta"):
        if not isinstance(deploy, self.deploy_type):
            raise ValueError(
                f"Meta of the {self.type} deployment should be {self.deploy_type}, not {deploy.__class__}"
            )


class DeployStatus(Enum):
    UNKNOWN = "unknown"
    NOT_DEPLOYED = "not_deployed"
    STARTING = "starting"
    CRASHED = "crashed"
    STOPPED = "stopped"
    RUNNING = "running"


class DeployMeta(MlemMeta):
    object_type: ClassVar = "deployment"

    class Config:
        type_root = True
        type_field = "type"
        exclude = {"model", "env"}
        use_enum_values = True

    abs_name: ClassVar = "deploy"
    type: ClassVar[str]

    env_link: MlemLink
    env: Optional[TargetEnvMeta]
    model_link: MlemLink
    model: Optional[ModelMeta]
    state: Optional[DeployState]

    def get_env(self):
        if self.env is None:
            self.env = self.env_link.bind(self.loc).load_link(
                force_type=TargetEnvMeta
            )
        return self.env

    def get_model(self):
        if self.model is None:
            self.model = self.model_link.bind(self.loc).load_link(
                force_type=ModelMeta
            )
        return self.model

    def deploy(self):
        return self.get_env().deploy(self)

    def destroy(self):
        self.get_env().destroy(self)

    def get_status(self, raise_on_error: bool = True) -> DeployStatus:
        return self.get_env().get_status(self, raise_on_error=raise_on_error)

    def wait_for_status(
        self,
        status: Union[DeployStatus, Iterable[DeployStatus]],
        timeout: float = 1.0,
        times: int = 5,
        allowed_intermediate: Union[
            DeployStatus, Iterable[DeployStatus]
        ] = None,
        raise_on_timeout: bool = True,
    ):
        if isinstance(status, DeployStatus):
            statuses = {status}
        else:
            statuses = set(status)
        allowed_intermediate = allowed_intermediate or set()
        if isinstance(allowed_intermediate, DeployStatus):
            allowed = {allowed_intermediate}
        else:
            allowed = set(allowed_intermediate)

        current = DeployStatus.UNKNOWN
        for _ in range(times):
            current = self.get_status(raise_on_error=False)
            if current in statuses:
                return True
            if allowed and current not in allowed:
                if raise_on_timeout:
                    raise DeploymentError(
                        f"Deployment status {current} is not allowed"
                    )
                return False
            time.sleep(timeout)
        if raise_on_timeout:
            raise DeploymentError(
                f"Deployment status is still {current} after {times * timeout} seconds"
            )
        return False


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
    META_FILE_NAME = "asdasdasdadassdas"
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
    type_, source_path = source_paths[0]
    return type_, source_path
