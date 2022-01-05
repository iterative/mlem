import os

import pytest

from mlem.core.errors import MlemRootNotFound
from mlem.utils.root import find_repo_root


def test_find_root(mlem_repo):
    path = os.path.join(mlem_repo, "subdir", "subdir")
    os.makedirs(path, exist_ok=True)
    assert find_repo_root(path) == mlem_repo


def test_find_root_error():
    path = os.path.dirname(__file__)
    with pytest.raises(MlemRootNotFound):
        find_repo_root(path, raise_on_missing=True)
    assert find_repo_root(path, raise_on_missing=False) is None


def test_find_root_strict(mlem_repo):
    assert find_repo_root(mlem_repo, recursive=False) == mlem_repo
    with pytest.raises(MlemRootNotFound):
        find_repo_root(os.path.join(mlem_repo, "subdir"), recursive=False)
