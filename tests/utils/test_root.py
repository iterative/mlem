import os

import pytest

from mlem.core.errors import MlemRootNotFound
from mlem.utils.root import find_mlem_root


def test_find_root(mlem_root):
    path = os.path.join(mlem_root, "subdir", "subdir")
    os.makedirs(path, exist_ok=True)
    assert find_mlem_root(path) == mlem_root


def test_find_root_error():
    path = os.path.dirname(__file__)
    with pytest.raises(MlemRootNotFound):
        find_mlem_root(path, raise_on_missing=True)
    assert find_mlem_root(path, raise_on_missing=False) is None


def test_find_root_strict(mlem_root):
    assert find_mlem_root(mlem_root, recursive=False) == mlem_root
    with pytest.raises(MlemRootNotFound):
        find_mlem_root(os.path.join(mlem_root, "subdir"), recursive=False)
