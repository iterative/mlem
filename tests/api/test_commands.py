import os

import pytest
from numpy import ndarray

from mlem.api import apply, link, load_meta
from mlem.core.meta_io import MLEM_DIR, MLEM_EXT
from mlem.core.objects import MlemLink, ModelMeta


@pytest.mark.parametrize(
    "m, d",
    [
        (
            pytest.lazy_fixture("model_meta"),
            pytest.lazy_fixture("dataset_meta"),
        ),
        (
            pytest.lazy_fixture("model_meta_saved"),
            pytest.lazy_fixture("dataset_meta_saved"),
        ),
        (pytest.lazy_fixture("model_meta_saved"), pytest.lazy_fixture("X")),
        (pytest.lazy_fixture("model_path"), pytest.lazy_fixture("data_path")),
    ],
)
def test_apply(m, d, request):
    res = apply(m, d, method="predict")
    assert isinstance(res, ndarray)


def test_link_as_separate_file(model_path_mlem_root):
    model_path, mlem_root = model_path_mlem_root
    link_path = os.path.join(mlem_root, "latest.mlem.yaml")
    link(model_path, target=link_path, mlem_root=None)
    assert os.path.exists(link_path)
    link_object = load_meta(link_path, follow_links=False)
    assert isinstance(link_object, MlemLink)
    model = load_meta(link_path)
    assert isinstance(model, ModelMeta)


def test_link_in_mlem_dir(model_path_mlem_root):
    model_path, mlem_root = model_path_mlem_root
    link_name = "latest"
    link_obj = link(model_path, target=link_name, mlem_root=mlem_root)
    assert isinstance(link_obj, MlemLink)
    link_dumped_to = os.path.join(
        mlem_root, MLEM_DIR, "model", link_name + MLEM_EXT
    )
    assert os.path.exists(link_dumped_to)
    loaded_link_object = load_meta(link_dumped_to, follow_links=False)
    assert isinstance(loaded_link_object, MlemLink)
    model = load_meta(link_dumped_to)
    assert isinstance(model, ModelMeta)
