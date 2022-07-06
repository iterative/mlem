"""
Base classes for meta objects in MLEM
"""
import hashlib
import os
import posixpath
import time
from abc import ABC, abstractmethod
from enum import Enum
from functools import partial
from typing import (
    Any,
    ClassVar,
    Dict,
    Generic,
    Iterable,
    Iterator,
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
from pydantic import ValidationError, parse_obj_as, validator
from typing_extensions import Literal
from yaml import safe_dump, safe_load

from mlem.config import project_config
from mlem.core.artifacts import (
    Artifacts,
    FSSpecStorage,
    LocalArtifact,
    PlaceholderArtifact,
)
from mlem.core.base import MlemABC
from mlem.core.data_type import DataReader, DataType
from mlem.core.errors import (
    DeploymentError,
    MlemObjectNotFound,
    MlemObjectNotSavedError,
    MlemProjectNotFound,
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
from mlem.ui import EMOJI_LINK, EMOJI_LOAD, EMOJI_SAVE, echo, no_echo
from mlem.utils.path import make_posix
from mlem.utils.root import find_project_root

T = TypeVar("T", bound="MlemObject")


class MlemObject(MlemABC):
    """Base class for MLEM objects.
    MLEM objects contain metadata about different types of objects and are saved
    in a form of `.mlem` files.
    """

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
    labels: List[str] = []

    @property
    def loc(self) -> Location:
        if self.location is None:
            raise MlemObjectNotSavedError("Not saved object has no location")
        return self.location

    @property
    def name(self):
        """Name of the object in the project"""
        project_path = self.loc.path_in_project[: -len(MLEM_EXT)]
        prefix = posixpath.join(MLEM_DIR, self.object_type)
        if project_path.startswith(prefix):
            project_path = project_path[len(prefix) + 1 :]
        return project_path

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
        project: Optional[str],
        fs: Optional[AbstractFileSystem],
        external: bool,
        ensure_mlem_root: bool,
        metafile_path: bool = True,
    ) -> Location:
        """Create location from arguments"""
        if metafile_path:
            path = cls.get_metafile_path(path)
        loc = UriResolver.resolve(
            path, project, rev=None, fs=fs, find_project=True
        )
        if loc.project is not None:
            # check that project is mlem project root
            find_project_root(
                loc.project, loc.fs, raise_on_missing=True, recursive=False
            )
        if ensure_mlem_root and loc.project is None:
            raise MlemProjectNotFound(loc.fullpath, loc.fs)
        if (
            loc.project is None
            or external
            or loc.fullpath.startswith(
                posixpath.join(loc.project, MLEM_DIR, cls.object_type)
            )
        ):
            # orphan or external or inside .mlem
            return loc

        internal_path = posixpath.join(
            MLEM_DIR,
            cls.object_type,
            loc.path_in_project,
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
        echo(
            EMOJI_LOAD
            + f"Loading {getattr(cls, 'object_type', 'meta')} from {location.uri_repr}"
        )
        with location.open() as f:
            payload = safe_load(f)
        res = parse_obj_as(MlemObject, payload).bind(location)
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
        project: Optional[str] = None,
        index: Optional[bool] = None,
        external: Optional[bool] = None,
    ):
        """Dumps metafile and possible artifacts to path.

        Args:
            path: name of the object. Relative to project, if it is provided.
            fs: filesystem to save to. if not provided, inferred from project and path
            project: path to mlem project
            index: whether add to index if object is external.
                If set to True, checks existanse of mlem project
                defaults to True if mlem project exists and external is true
            external: whether to save object inside mlem dir or not.
                Defaults to false if project is provided
                Forced to false if path points inside mlem dir
        """
        location, index = self._parse_dump_args(
            path, project, fs, index, external
        )
        self._write_meta(location, index)
        return self

    def _write_meta(
        self,
        location: Location,
        index: bool,
    ):
        """Write metadata to path in fs and possibly create link in mlem dir"""
        echo(EMOJI_SAVE + f"Saving {self.object_type} to {location.uri_repr}")
        location.fs.makedirs(
            posixpath.dirname(location.fullpath), exist_ok=True
        )
        with location.open("w") as f:
            safe_dump(self.dict(), f)
        if index and location.project:
            project_config(location.project, location.fs).index.index(
                self, location
            )

    def _parse_dump_args(
        self,
        path: str,
        project: Optional[str],
        fs: Optional[AbstractFileSystem],
        index: Optional[bool],
        external: Optional[bool],
    ) -> Tuple[Location, bool]:
        """Parse arguments for .dump and bind meta"""
        if external is None:
            external = project_config(project, fs=fs).EXTERNAL
        # by default we index only external non-orphan objects
        if index is None:
            index = True
            ensure_mlem_root = False
        else:
            # if index manually set to True, there should be mlem project
            ensure_mlem_root = index
        location = self._get_location(
            make_posix(path),
            make_posix(project),
            fs,
            external,
            ensure_mlem_root,
        )
        self.bind(location)
        if location.project is not None:
            # force external=False if fullpath inside MLEM_DIR
            external = posixpath.join(MLEM_DIR, "") not in posixpath.dirname(
                location.fullpath
            )
        return location, index

    def make_link(
        self,
        path: str = None,
        fs: Optional[AbstractFileSystem] = None,
        project: Optional[str] = None,
        external: Optional[bool] = None,
        absolute: bool = False,
    ) -> "MlemLink":
        if self.location is None:
            raise MlemObjectNotSavedError(
                "Cannot create link for not saved meta object"
            )
        link = MlemLink(
            path=self.loc.path,
            project=self.loc.project_uri,
            rev=self.loc.rev,
            link_type=self.resolved_type,
        )
        if path is not None:
            (
                location,
                _,
            ) = link._parse_dump_args(  # pylint: disable=protected-access
                path, project, fs, False, external=external
            )
            if (
                not absolute
                and self.loc.is_same_project(location)
                and self.loc.rev is None
            ):
                link.path = self.get_metafile_path(self.name)
                link.link_type = self.resolved_type
                link.project = None
            link._write_meta(  # pylint: disable=protected-access
                location, True
            )
        return link

    def clone(
        self,
        path: str,
        fs: Union[str, AbstractFileSystem, None] = None,
        project: Optional[str] = None,
        index: Optional[bool] = None,
        external: Optional[bool] = None,
    ):
        """
        Clone existing object to `path`.

        Arguments are the same as for `dump`
        """
        if not self.is_saved:
            raise MlemObjectNotSavedError("Cannot clone not saved object")
        new: "MlemObject" = self.deepcopy()
        new.dump(
            path, fs, project, index, external
        )  # only dump meta TODO: https://github.com/iterative/mlem/issues/37
        return new

    def deepcopy(self):
        return parse_obj_as(
            MlemObject, self.dict()
        )  # easier than deep copy bc of possible attached objects

    def update(self):
        if not self.is_saved:
            raise MlemObjectNotSavedError("Cannot update not saved object")
        echo(
            EMOJI_SAVE
            + f"Updating {self.object_type} at {self.location.uri_repr}"
        )
        with no_echo():
            self._write_meta(self.location, False)

    def meta_hash(self):
        return hashlib.md5(safe_dump(self.dict()).encode("utf8")).hexdigest()


class MlemLink(MlemObject):
    """Link is a special MlemObject that represents a MlemObject in a different
    location"""

    path: str
    project: Optional[str] = None
    rev: Optional[str] = None
    link_type: str

    object_type: ClassVar = "link"

    @property
    def link_cls(self) -> Type[MlemObject]:
        return MlemObject.__type_map__[self.link_type]

    @property
    def resolved_type(self):
        return self.link_type

    @validator("path", "project", allow_reuse=True)
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
    ) -> MlemObject:
        ...

    def load_link(
        self, follow_links: bool = True, *, force_type: Type[MlemObject] = None
    ) -> MlemObject:
        if force_type is not None and self.link_cls != force_type:
            raise WrongMetaType(self.link_type, force_type)
        link = self.parse_link()
        echo(EMOJI_LINK + f"Loading link to {link.uri_repr}")
        with no_echo():
            return self.link_cls.read(link, follow_links=follow_links)

    def parse_link(self) -> Location:
        from mlem.core.metadata import find_meta_location

        if self.project is None and self.rev is None:
            # is it possible to have rev without project?
            location = UriResolver.resolve(
                path=self.path, project=None, rev=None, fs=None
            )
            if (
                location.project is None
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
                path=self.path, project=self.project, rev=self.rev, fs=None
            )
        )

    @classmethod
    def from_location(
        cls, loc: Location, link_type: Union[str, Type[MlemObject]]
    ) -> "MlemLink":
        return MlemLink(
            path=get_path_by_fs_path(loc.fs, loc.path_in_project),
            project=loc.project,
            rev=loc.rev,
            link_type=link_type.object_type
            if not isinstance(link_type, str)
            else link_type,
        )


class _WithArtifacts(ABC, MlemObject):
    """Special subtype of MlemObject that can have files (artifacts) attached"""

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
        project_path = self.location.path_in_project
        prefix = posixpath.join(MLEM_DIR, self.object_type)
        if project_path.startswith(prefix):
            project_path = project_path[len(prefix) + 1 :]
        if project_path.endswith(MLEM_EXT):
            project_path = project_path[: -len(MLEM_EXT)]
        return project_path

    @property
    def basename(self):
        res = posixpath.basename(self.location.path)
        if res.endswith(MLEM_EXT):
            res = res[: -len(MLEM_EXT)]
        return res

    @property
    def path(self):
        path = self.location.fullpath
        if path.endswith(MLEM_EXT):
            path = path[: -len(MLEM_EXT)]
        return path

    def dump(
        self,
        path: str,
        fs: Union[str, AbstractFileSystem, None] = None,
        project: Optional[str] = None,
        index: Optional[bool] = None,
        external: Optional[bool] = None,
    ):
        location, index = self._parse_dump_args(
            path, project, fs, index, external
        )
        try:
            if location.exists():
                with no_echo():
                    existing = MlemObject.read(location, follow_links=False)
                if isinstance(existing, _WithArtifacts):
                    for art in existing.relative_artifacts.values():
                        art.remove()
        except (MlemObjectNotFound, FileNotFoundError, ValidationError):
            pass
        self.artifacts = self.get_artifacts()
        self._write_meta(location, index)
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
        project: Optional[str] = None,
        index: Optional[bool] = None,
        external: Optional[bool] = None,
    ):
        if self.location is None:
            raise MlemObjectNotSavedError("Cannot clone not saved object")
        # clone is just dump with copying artifacts
        new: "_WithArtifacts" = self.deepcopy()
        new.artifacts = {}
        (
            location,
            index,
        ) = new._parse_dump_args(  # pylint: disable=protected-access
            path, project, fs, index, external
        )

        for art_name, art in (self.artifacts or {}).items():
            if isinstance(art, LocalArtifact) and not posixpath.isabs(art.uri):
                art_path = new.path + art.uri[len(self.basename) :]
            else:
                art_path = posixpath.join(new.path, art_name)
            download = art.relative(
                self.location.fs, self.dirname
            ).materialize(art_path, new.loc.fs)
            new.artifacts[art_name] = LocalArtifact(
                uri=posixpath.relpath(art_path, new.dirname), **download.info
            )
        new._write_meta(location, index)  # pylint: disable=protected-access
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
        return {
            name: a.relative(self.location.fs, self.dirname)
            for name, a in (self.artifacts or {}).items()
        }

    @property
    def storage(self):
        if not self.location.fs or isinstance(
            self.location.fs, LocalFileSystem
        ):
            return project_config(
                self.loc.project, self.loc.fs
            ).storage.relative(self.location.fs, self.dirname)
        return FSSpecStorage.from_fs_path(self.location.fs, self.dirname)

    def get_artifacts(self):
        if self.artifacts is None:
            return self.write_value()
        return {
            name: a.relative_to(self.loc)
            if isinstance(a, PlaceholderArtifact)
            else a
            for name, a in self.artifacts.items()
        }

    def checkenv(self):
        self.requirements.check()


class MlemModel(_WithArtifacts):
    """MlemObject representing a ML model"""

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
        labels: List[str] = None,
        params: Dict[str, str] = None,
    ) -> "MlemModel":
        mt = ModelAnalyzer.analyze(model, sample_data=sample_data)
        if mt.model is None:
            mt = mt.bind(model)

        return MlemModel(
            model_type=mt,
            requirements=mt.get_requirements().expanded,
            description=description,
            labels=labels or [],
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


class MlemData(_WithArtifacts):
    """MlemObject representing data"""

    class Config:
        exclude = {"data"}

    object_type: ClassVar = "data"
    reader_cache: Optional[Dict]
    reader: Optional[DataReader]
    reader, reader_raw, reader_cache = lazy_field(
        DataReader,
        "reader",
        "reader_cache",
        parse_as_type=Optional[DataReader],
        default=None,
    )
    data_type: Optional[DataType] = None

    @property
    def data(self):
        return self.data_type.data

    @classmethod
    def from_data(
        cls,
        data: Any,
        description: str = None,
        params: Dict[str, str] = None,
        labels: List[str] = None,
    ) -> "MlemData":
        data_type = DataType.create(
            data,
        )
        meta = MlemData(
            requirements=data_type.get_requirements().expanded,
            description=description,
            params=params or {},
            labels=labels or [],
        )
        meta.data_type = data_type
        return meta

    def write_value(self) -> Artifacts:
        if self.data_type is not None:
            filename = os.path.basename(self.name)
            reader, artifacts = self.data_type.get_writer(
                project=self.loc.project, filename=filename
            ).write(
                self.data_type,
                self.storage,
                filename,
            )
            self.reader = reader
            return artifacts
        raise ValueError("Meta is not binded to actual data")

    def load_value(self):
        self.data_type = self.reader.read(self.relative_artifacts)

    def read_batch(self, batch_size: int) -> Iterator[DataType]:
        if self.reader is None:
            raise MlemObjectNotSavedError(
                "Cannot read batch from not saved data"
            )
        return self.reader.read_batch(self.relative_artifacts, batch_size)

    def get_value(self):
        return self.data


class MlemBuilder(MlemObject):
    """Base class to define different ways of building/exporting models
    into different formats"""

    class Config:
        type_root = True
        type_field = "type"

    object_type: ClassVar = "builder"
    abs_name: ClassVar[str] = "builder"

    @abstractmethod
    def build(self, obj: MlemModel):  # TODO maybe we can also pack datasets?
        raise NotImplementedError


class DeployState(MlemABC):
    """Base class for deployment state metadata"""

    class Config:
        type_root = True

    abs_name: ClassVar[str] = "deploy_state"

    model_hash: Optional[str] = None

    @abstractmethod
    def get_client(self):
        raise NotImplementedError


DT = TypeVar("DT", bound="MlemDeployment")


class MlemEnv(MlemObject, Generic[DT]):
    """Base class for target environment metadata"""

    class Config:
        type_root = True
        type_field = "type"

    abs_name = "env"
    object_type: ClassVar = "env"
    type: ClassVar = ...
    deploy_type: ClassVar[Type[DT]]

    @abstractmethod
    def deploy(self, meta: DT):
        raise NotImplementedError

    @abstractmethod
    def remove(self, meta: DT):
        raise NotImplementedError

    @abstractmethod
    def get_status(self, meta: DT, raise_on_error=True) -> "DeployStatus":
        raise NotImplementedError

    def check_type(self, deploy: "MlemDeployment"):
        if not isinstance(deploy, self.deploy_type):
            raise ValueError(
                f"Meta of the {self.type} deployment should be {self.deploy_type}, not {deploy.__class__}"
            )


class DeployStatus(str, Enum):
    """Enum with deployment statuses"""

    UNKNOWN = "unknown"
    NOT_DEPLOYED = "not_deployed"
    STARTING = "starting"
    CRASHED = "crashed"
    STOPPED = "stopped"
    RUNNING = "running"


class MlemDeployment(MlemObject):
    """Base class for deployment metadata"""

    object_type: ClassVar = "deployment"

    class Config:
        type_root = True
        type_field = "type"
        exclude = {"model", "env"}
        use_enum_values = True

    abs_name: ClassVar = "deployment"
    type: ClassVar[str]

    env_link: MlemLink
    env: Optional[MlemEnv]
    model_link: MlemLink
    model: Optional[MlemModel]
    state: Optional[DeployState]

    def get_env(self):
        if self.env is None:
            self.env = self.env_link.bind(self.loc).load_link(
                force_type=MlemEnv
            )
        return self.env

    def get_model(self):
        if self.model is None:
            self.model = self.model_link.bind(self.loc).load_link(
                force_type=MlemModel
            )
        return self.model

    def run(self):
        return self.get_env().deploy(self)

    def remove(self):
        self.get_env().remove(self)

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

    def model_changed(self):
        if self.state is None or self.state.model_hash is None:
            return True
        return self.get_model().meta_hash() != self.state.model_hash

    def update_model_hash(self, model: Optional[MlemModel] = None):
        model = model or self.get_model()
        if self.state is None:
            return
        self.state.model_hash = model.meta_hash()

    def replace_model(self, model: MlemModel):
        self.model = model
        self.model_link = self.model.make_link()


def find_object(
    path: str, fs: AbstractFileSystem, project: str = None
) -> Tuple[str, str]:
    """Extract object_type and path from path.
    assumes .mlem/ content is valid"""
    if project is None:
        project = find_project_root(path, fs)
    if project is not None and path.startswith(project):
        path = os.path.relpath(path, project)
    source_paths = [
        (
            tp,
            posixpath.join(
                project or "",
                MLEM_DIR,
                cls.object_type,
                cls.get_metafile_path(path),
            ),
        )
        for tp, cls in MlemObject.non_abstract_subtypes().items()
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
