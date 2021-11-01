# pylint: disable=no-member
import os

import pytest
from numpy import ndarray

from mlem.api import apply, link, load_meta
from mlem.api.commands import ls
from mlem.core.meta_io import MLEM_DIR, MLEM_EXT
from mlem.core.objects import DatasetMeta, MlemLink, ModelMeta
from tests.conftest import MLEM_TEST_REPO, long


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
        (
            pytest.lazy_fixture("model_meta_saved"),
            pytest.lazy_fixture("train"),
        ),
        (pytest.lazy_fixture("model_path"), pytest.lazy_fixture("data_path")),
    ],
)
def test_apply(m, d):
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


def test_ls_local(mlem_root):
    objects = ls(mlem_root)
    assert len(objects) == 1
    assert ModelMeta in objects
    models = objects[ModelMeta]
    assert len(models) == 2
    model, lnk = models
    if isinstance(model, MlemLink):
        model, lnk = lnk, model

    assert isinstance(model, ModelMeta)
    assert isinstance(lnk, MlemLink)
    assert os.path.join(mlem_root, lnk.mlem_link) == model.name


@long
def test_ls_remote():
    objects = ls(os.path.join(MLEM_TEST_REPO, "tree/main/simple"))
    assert len(objects) == 2
    assert ModelMeta in objects
    models = objects[ModelMeta]
    assert len(models) == 2
    model, lnk = models
    if isinstance(model, MlemLink):
        model, lnk = lnk, model

    assert isinstance(model, ModelMeta)
    assert isinstance(lnk, MlemLink)

    assert DatasetMeta in objects
    assert len(objects[DatasetMeta]) == 3
