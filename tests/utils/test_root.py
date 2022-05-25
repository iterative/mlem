import os

import pytest

from mlem.core.errors import MlemProjectNotFound
from mlem.utils.root import find_project_root


def test_find_root(mlem_project):
    path = os.path.join(mlem_project, "subdir", "subdir")
    os.makedirs(path, exist_ok=True)
    assert find_project_root(path) == mlem_project


def test_find_root_error():
    path = os.path.dirname(__file__)
    with pytest.raises(MlemProjectNotFound):
        find_project_root(path, raise_on_missing=True)
    assert find_project_root(path, raise_on_missing=False) is None


def test_find_root_strict(mlem_project):
    assert find_project_root(mlem_project, recursive=False) == mlem_project
    with pytest.raises(MlemProjectNotFound):
        find_project_root(
            os.path.join(mlem_project, "subdir"), recursive=False
        )
