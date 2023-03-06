import pytest
from pydantic import ValidationError
from yaml import safe_dump

from mlem.api.migrations import migrate
from mlem.core.metadata import load_meta
from mlem.core.objects import MlemModel, MlemObject

model_02 = (
    {
        "object_type": "model",
        "description": "machine learning should be mlemming",
        "labels": ["mlemming", "it", "should", "be"],
        "artifacts": {},
        "model_type": {"type": "sklearn", "methods": {"lol": {}}},
    },
    MlemModel(
        artifacts={},
        call_orders={"lol": [("model", "lol")]},
        processors={"model": {"type": "sklearn", "methods": {"lol": {}}}},
    ),
)


model_03 = (
    {
        "object_type": "model",
        "artifacts": {},
        "model_type": {"type": "sklearn", "methods": {"lol": {}}},
    },
    MlemModel(
        artifacts={},
        call_orders={"lol": [("model", "lol")]},
        processors={"model": {"type": "sklearn", "methods": {"lol": {}}}},
    ),
)


@pytest.mark.parametrize("old_data", [model_02, model_03])
def test_single(tmpdir, old_data):
    path = tmpdir / "model.mlem"
    old_payload, new_object = old_data
    path.write_text(safe_dump(old_payload), encoding="utf8")

    migrate(str(path))

    meta = load_meta(path, try_migrations=False)

    assert isinstance(meta, MlemObject)
    assert meta == new_object


@pytest.mark.parametrize("old_data,new_data", [model_02, model_03])
@pytest.mark.parametrize("recursive", [True, False])
def test_directory(tmpdir, old_data, new_data, recursive):
    subdir_path = tmpdir / "subdir" / "model.mlem"
    (tmpdir / "subdir").mkdir()
    subdir_path.write_text(safe_dump(old_data), encoding="utf8")
    for i in range(3):
        path = tmpdir / f"model{i}.mlem"
        path.write_text(safe_dump(old_data), encoding="utf8")

    migrate(str(tmpdir), recursive=recursive)

    for i in range(3):
        path = tmpdir / f"model{i}.mlem"
        meta = load_meta(path, try_migrations=False)
        assert isinstance(meta, MlemObject)
        assert meta == new_data

    if recursive:
        meta = load_meta(subdir_path, try_migrations=False)
        assert isinstance(meta, MlemObject)
        assert meta == new_data
    else:
        try:
            assert load_meta(subdir_path, try_migrations=False) != new_data
        except ValidationError:
            pass


@pytest.mark.parametrize("old_data,new_data", [model_02, model_03])
def test_load_with_migration(tmpdir, old_data, new_data):
    path = tmpdir / "model.mlem"
    path.write_text(safe_dump(old_data), encoding="utf8")

    meta = load_meta(path, try_migrations=True)

    assert isinstance(meta, MlemObject)
    assert meta == new_data
