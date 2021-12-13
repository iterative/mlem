# pylint: disable=no-member
import os
import posixpath

import pytest
from numpy import ndarray

from mlem.api import apply, link, load_meta
from mlem.api.commands import init, ls
from mlem.config import CONFIG_FILE
from mlem.core.errors import MlemRootNotFound
from mlem.core.meta_io import MLEM_DIR, MLEM_EXT
from mlem.core.objects import DatasetMeta, MlemLink, ModelMeta
from mlem.utils.path import make_posix
from tests.conftest import MLEM_TEST_REPO, issue_110, long, need_test_repo_auth


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


def test_link_as_separate_file(model_path_mlem_repo):
    model_path, mlem_repo = model_path_mlem_repo
    link_path = os.path.join(mlem_repo, "latest.mlem.yaml")
    link(model_path, target=link_path, external=True)
    assert os.path.exists(link_path)
    link_object = load_meta(link_path, follow_links=False)
    assert isinstance(link_object, MlemLink)
    model = load_meta(link_path)
    assert isinstance(model, ModelMeta)


def test_link_in_mlem_dir(model_path_mlem_repo):
    model_path, mlem_repo = model_path_mlem_repo
    link_name = "latest"
    link_obj = link(
        model_path,
        target=link_name,
        target_repo=mlem_repo,
        external=False,
    )
    assert isinstance(link_obj, MlemLink)
    link_dumped_to = os.path.join(
        mlem_repo, MLEM_DIR, "link", link_name + MLEM_EXT
    )
    assert os.path.exists(link_dumped_to)
    loaded_link_object = load_meta(link_dumped_to, follow_links=False)
    assert isinstance(loaded_link_object, MlemLink)
    model = load_meta(link_dumped_to)
    assert isinstance(model, ModelMeta)


def test_ls_local(filled_mlem_repo):
    objects = ls(filled_mlem_repo)
    assert len(objects) == 1
    assert ModelMeta in objects
    models = objects[ModelMeta]
    assert len(models) == 2
    model, lnk = models
    if isinstance(model, MlemLink):
        model, lnk = lnk, model

    assert isinstance(model, ModelMeta)
    assert isinstance(lnk, MlemLink)
    assert (
        posixpath.join(make_posix(filled_mlem_repo), lnk.path)
        == model.loc.fullpath
    )


def test_ls_no_repo(tmpdir):
    with pytest.raises(MlemRootNotFound):
        ls(str(tmpdir))


@long
@need_test_repo_auth
def test_ls_remote(current_test_branch):
    objects = ls(
        os.path.join(MLEM_TEST_REPO, f"tree/{current_test_branch}/simple")
    )
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


def test_init(tmpdir):
    init(str(tmpdir))
    assert os.path.isdir(tmpdir / MLEM_DIR)
    assert os.path.isfile(tmpdir / MLEM_DIR / CONFIG_FILE)


@long
@issue_110
def test_init_remote(s3_tmp_path, s3_storage_fs):
    path = s3_tmp_path("init")
    init(path)
    assert s3_storage_fs.isdir(f"{path}/{MLEM_DIR}")
    assert s3_storage_fs.isfile(f"{path}/{MLEM_DIR}/{CONFIG_FILE}")
