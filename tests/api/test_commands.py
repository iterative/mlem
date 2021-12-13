import os
import pickle
import posixpath

import pytest
from numpy import ndarray
from pytest_lazyfixture import lazy_fixture

from mlem.api import apply, link, load_meta
from mlem.api.commands import import_path, init, ls
from mlem.config import CONFIG_FILE
from mlem.core.artifacts import LocalArtifact
from mlem.core.meta_io import ART_DIR, META_FILE_NAME, MLEM_DIR, MLEM_EXT
from mlem.core.metadata import load
from mlem.core.model import SimplePickleIO
from mlem.core.objects import DatasetMeta, MlemLink, ModelMeta
from mlem.utils.path import make_posix
from tests.conftest import MLEM_TEST_REPO, issue_110, long, need_test_repo_auth


@pytest.mark.parametrize(
    "m, d",
    [
        (
            lazy_fixture("model_meta"),
            lazy_fixture("dataset_meta"),
        ),
        (
            lazy_fixture("model_meta_saved"),
            lazy_fixture("dataset_meta_saved"),
        ),
        (
            lazy_fixture("model_meta_saved"),
            lazy_fixture("train"),
        ),
        (lazy_fixture("model_path"), lazy_fixture("data_path")),
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


@pytest.mark.parametrize("file_ext, type_", [(".pkl", None), ("", "pickle")])
def test_import_model_pickle__move(model, train, tmpdir, file_ext, type_):
    path = str(tmpdir / "mymodel" + file_ext)
    with open(path, "wb") as f:
        pickle.dump(model, f)

    out_path = str(tmpdir / "mlem_model")
    meta = import_path(path, out=out_path, type_=type_, move=True)
    assert isinstance(meta, ModelMeta)
    assert os.path.isdir(out_path)
    assert os.path.isfile(os.path.join(out_path, META_FILE_NAME))
    assert os.path.isdir(os.path.join(out_path, ART_DIR))
    assert os.path.isfile(
        os.path.join(out_path, ART_DIR, SimplePickleIO.file_name)
    )
    loaded = load(out_path)
    loaded.predict(train)


@pytest.mark.parametrize("file_ext, type_", [(".pkl", None), ("", "pickle")])
def test_import_model_pickle__no_move(model, train, tmpdir, file_ext, type_):
    path = str(tmpdir / "mymodel" + file_ext)
    with open(path, "wb") as f:
        pickle.dump(model, f)

    out_path = str(tmpdir / "mlem_model")
    meta = import_path(path, out=out_path, type_=type_, move=False)
    assert isinstance(meta, ModelMeta)
    assert os.path.isdir(out_path)
    assert os.path.isfile(os.path.join(out_path, META_FILE_NAME))
    assert not os.path.exists(os.path.join(out_path, ART_DIR))
    loaded_meta = load_meta(out_path, load_value=True)
    assert isinstance(loaded_meta, ModelMeta)
    assert len(loaded_meta.artifacts) == 1
    art = loaded_meta.artifacts[0]
    assert loaded_meta.loc.fs.exists(art.uri)
    loaded = loaded_meta.get_value()
    loaded.predict(train)


@pytest.mark.parametrize("file_ext, type_", [(".pkl", None), ("", "pickle")])
def test_import_model_pickle__no_move_in_mlem_repo(
    model, train, mlem_repo, file_ext, type_
):
    filename = "mymodel" + file_ext
    path = os.path.join(mlem_repo, filename)
    with open(path, "wb") as f:
        pickle.dump(model, f)

    out_path = os.path.join(mlem_repo, "mlem_model")
    meta = import_path(
        path, out=out_path, type_=type_, move=False, external=True
    )
    assert isinstance(meta, ModelMeta)
    assert os.path.isdir(out_path)
    assert os.path.isfile(os.path.join(out_path, META_FILE_NAME))
    assert not os.path.exists(os.path.join(out_path, ART_DIR))
    loaded_meta = load_meta(out_path, load_value=True)
    assert isinstance(loaded_meta, ModelMeta)
    assert len(loaded_meta.artifacts) == 1
    art = loaded_meta.artifacts[0]
    assert isinstance(art, LocalArtifact)
    assert art.uri == f"../{filename}"
    loaded = loaded_meta.get_value()
    loaded.predict(train)
