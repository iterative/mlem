import posixpath
from typing import Callable, List, Optional

from yaml import safe_dump, safe_load

from mlem.core.errors import MlemObjectNotFound
from mlem.core.meta_io import MLEM_EXT, Location
from mlem.core.metadata import find_meta_location
from mlem.ui import echo
from mlem.utils.path import make_posix


def migrate(path: str, project: Optional[str] = None, recursive: bool = False):
    path = posixpath.join(make_posix(project or ""), make_posix(path))
    location = Location.resolve(path)
    try:
        location = find_meta_location(location)
        _migrate_one(location)
        return
    except MlemObjectNotFound:
        pass

    postfix = f"/**{MLEM_EXT}" if recursive else f"/*{MLEM_EXT}"
    for filepath in location.fs.glob(
        location.fullpath + postfix, detail=False
    ):
        print(filepath)
        loc = location.copy()
        loc.update_path(filepath)
        _migrate_one(loc)


def apply_migrations(payload: dict):
    changed = False
    for migration in _migrations:
        migrated = migration(payload)
        if migrated is not None:
            payload = migrated
            changed = True
    return payload, changed


def _migrate_one(location: Location):
    with location.open("r") as f:
        payload = safe_load(f)

    payload, changed = apply_migrations(payload)

    if changed:
        echo(f"Migrated MLEM Object at {location}")
        with location.open("w") as f:
            safe_dump(payload, f)


def _migrate_to_028(meta: dict) -> Optional[dict]:
    if "object_type" not in meta:
        return None

    if "description" in meta:
        meta.pop("description")

    if "labels" in meta:
        meta.pop("labels")
    return meta


def _migrate_to_040(meta: dict) -> Optional[dict]:
    if "object_type" not in meta or meta["object_type"] != "model":
        return None

    if "model_type" not in meta:
        return None

    main_model = meta.pop("model_type")
    meta["processors"] = {"model": main_model}
    meta["call_orders"] = {
        method: [("model", method)] for method in main_model["methods"]
    }
    return meta


_migrations: List[Callable[[dict], Optional[dict]]] = [
    _migrate_to_028,
    _migrate_to_040,
]
