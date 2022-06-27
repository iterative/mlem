import posixpath
from abc import abstractmethod
from collections import defaultdict
from typing import ClassVar, Dict, Iterable, List, Set, Type, Union

from pydantic import ValidationError, parse_obj_as
from yaml import safe_dump, safe_load

from mlem.constants import MLEM_DIR
from mlem.core.base import MlemABC
from mlem.core.errors import MlemProjectNotFound
from mlem.core.meta_io import MLEM_EXT, Location
from mlem.core.metadata import load_meta
from mlem.core.objects import MlemLink, MlemObject
from mlem.ui import no_echo

TypeFilter = Union[Type[MlemObject], Iterable[Type[MlemObject]], None]


class Index(MlemABC):
    """Base class for mlem object indexing logic"""

    class Config:
        type_root = True

    abs_name: ClassVar = "index"

    @abstractmethod
    def index(self, obj: MlemObject, location: Location):
        raise NotImplementedError

    @abstractmethod
    def list(
        self,
        location: Location,
        type_filter: TypeFilter,
        include_links: bool = True,
    ) -> Dict[Type[MlemObject], List[MlemObject]]:
        raise NotImplementedError

    @staticmethod
    def parse_type_filter(type_filter: TypeFilter) -> Set[Type[MlemObject]]:
        if type_filter is None:
            type_filter = set(MlemObject.non_abstract_subtypes().values())
        if isinstance(type_filter, type) and issubclass(
            type_filter, MlemObject
        ):
            type_filter = {type_filter}
        tf = set(type_filter)
        if not tf:
            return set()
        tf.add(MlemLink)
        return tf


class LinkIndex(Index):
    """Indexing base on contents of MLEM_DIR - either objects or links to them
    should be there"""

    type: ClassVar = "link"

    def index(self, obj: MlemObject, location: Location):
        if (
            location.path
            == posixpath.join(MLEM_DIR, obj.object_type, obj.name) + MLEM_EXT
        ):
            return
        with no_echo():
            obj.make_link(
                obj.name, location.fs, project=location.project, external=False
            )

    def list(
        self,
        location: Location,
        type_filter: TypeFilter,
        include_links: bool = True,
        ignore_errors: bool = False,
    ) -> Dict[Type[MlemObject], List[MlemObject]]:
        _type_filter = self.parse_type_filter(type_filter)
        if len(_type_filter) == 0:
            return {}

        res = defaultdict(list)
        root_path = posixpath.join(location.project or "", MLEM_DIR)
        files = location.fs.glob(
            posixpath.join(root_path, f"**{MLEM_EXT}"), recursive=True
        )
        for cls in _type_filter:
            type_path = posixpath.join(root_path, cls.object_type)
            for file in files:
                if not file.startswith(type_path):
                    continue
                try:
                    with no_echo():
                        meta = load_meta(
                            posixpath.relpath(file, location.project),
                            project=location.project,
                            rev=location.rev,
                            follow_links=False,
                            fs=location.fs,
                            load_value=False,
                        )
                    obj_type = cls
                    if isinstance(meta, MlemLink):
                        link_name = posixpath.relpath(file, type_path)[
                            : -len(MLEM_EXT)
                        ]
                        is_auto_link = meta.path == link_name + MLEM_EXT

                        obj_type = MlemObject.__type_map__[meta.link_type]
                        if obj_type not in _type_filter:
                            continue
                        if is_auto_link:
                            with no_echo():
                                meta = meta.load_link()
                        elif not include_links:
                            continue
                    res[obj_type].append(meta)
                except ValidationError:
                    if not ignore_errors:
                        raise
        return res


FileIndexSchema = Dict[str, List[str]]


class FileIndex(Index):
    """Index as a single file"""

    type: ClassVar = "file"
    filename = "index.yaml"

    def _read_index(self, location: Location):
        if location.project is None:
            raise MlemProjectNotFound(location.path, location.fs, location.rev)
        path = posixpath.join(location.project, MLEM_DIR, self.filename)
        if not location.fs.exists(path):
            return {}

        with location.fs.open(path) as f:
            return parse_obj_as(FileIndexSchema, safe_load(f))

    def _write_index(self, location: Location, data: FileIndexSchema):
        if location.project is None:
            raise MlemProjectNotFound(location.path, location.fs, location.rev)
        path = posixpath.join(location.project, MLEM_DIR, self.filename)

        with location.fs.open(path, "w") as f:
            safe_dump(data, f)

    def index(self, obj: MlemObject, location: Location):
        data = self._read_index(location)
        type_data = data.get(obj.object_type, [])
        if obj.name not in type_data:
            type_data.append(obj.name)
            data[obj.object_type] = type_data
            self._write_index(location, data)

    def list(
        self,
        location: Location,
        type_filter: TypeFilter,
        include_links: bool = True,
    ) -> Dict[Type[MlemObject], List[MlemObject]]:
        _type_filter = self.parse_type_filter(type_filter)
        if not _type_filter:
            return {}

        data = self._read_index(location)

        res = defaultdict(list)

        with no_echo():
            for type_ in _type_filter:
                if type_ is MlemLink and not include_links:
                    continue

                res[type_].extend(
                    [
                        load_meta(
                            path,
                            location.project,
                            location.rev,
                            load_value=False,
                            fs=location.fs,
                        )
                        for path in data.get(type_.object_type, [])
                    ]
                )
            return res
