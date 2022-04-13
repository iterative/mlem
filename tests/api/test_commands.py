import os
import pickle
import posixpath

import pytest
from fsspec.implementations.local import LocalFileSystem
from numpy import ndarray
from pytest_lazyfixture import lazy_fixture

from mlem.api import apply, link, load_meta
from mlem.api.commands import import_object, init, ls
from mlem.config import CONFIG_FILE_NAME
from mlem.core.artifacts import LocalArtifact
from mlem.core.errors import MlemRootNotFound
from mlem.core.meta_io import MLEM_DIR, MLEM_EXT
from mlem.core.metadata import load
from mlem.core.model import ModelIO
from mlem.core.objects import DatasetMeta, MlemLink, ModelMeta
from mlem.utils.path import make_posix
from tests.conftest import MLEM_TEST_REPO, long, need_test_repo_auth

IMPORT_MODEL_FILENAME = "mymodel"


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
    link_path = os.path.join(mlem_repo, "latest.mlem")
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
    assert loaded_link_object.repo is None
    assert loaded_link_object.rev is None
    assert (
        loaded_link_object.path
        == os.path.relpath(model_path, mlem_repo) + MLEM_EXT
    )
    model = load_meta(link_dumped_to)
    assert isinstance(model, ModelMeta)


@long
def test_link_from_remote_to_local(current_test_branch, mlem_repo):
    link(
        "simple/data/model",
        source_repo=MLEM_TEST_REPO,
        rev="main",
        target="remote",
        target_repo=mlem_repo,
    )
    loaded_link_object = load_meta(
        "remote", repo=mlem_repo, follow_links=False
    )
    assert isinstance(loaded_link_object, MlemLink)
    assert loaded_link_object.repo == MLEM_TEST_REPO
    assert loaded_link_object.rev == "main"
    assert loaded_link_object.path == "simple/data/model" + MLEM_EXT
    model = loaded_link_object.load_link()
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
    assert os.path.isfile(tmpdir / MLEM_DIR / CONFIG_FILE_NAME)


@long
def test_init_remote(s3_tmp_path, s3_storage_fs):
    path = s3_tmp_path("init")
    init(path)
    assert s3_storage_fs.isdir(f"{path}/{MLEM_DIR}")
    assert s3_storage_fs.isfile(f"{path}/{MLEM_DIR}/{CONFIG_FILE_NAME}")


def _check_meta(meta, out_path, fs=None):
    assert isinstance(meta, ModelMeta)
    fs = fs or LocalFileSystem()
    assert fs.isfile(out_path + MLEM_EXT)


@pytest.fixture
def write_model_pickle(model):
    def write(path, fs=None):
        fs = fs or LocalFileSystem()
        with fs.open(path, "wb") as f:
            pickle.dump(model, f)

    return write


@pytest.mark.parametrize("file_ext, type_", [(".pkl", None), ("", "pickle")])
def test_import_model_pickle_copy(
    write_model_pickle, train, tmpdir, file_ext, type_
):
    path = str(tmpdir / "mymodel" + file_ext)
    write_model_pickle(path)

    out_path = str(tmpdir / "mlem_model")
    meta = import_object(path, target=out_path, type_=type_, copy_data=True)
    _check_meta(meta, out_path)
    assert os.path.isfile(out_path)
    loaded = load(out_path)
    loaded.predict(train)


def _check_load_artifact(
    meta,
    out_path,
    is_abs,
    train,
    filename=IMPORT_MODEL_FILENAME,
):
    assert isinstance(meta, ModelMeta)
    assert len(meta.artifacts) == 1
    art = meta.artifacts[ModelIO.art_name]
    if is_abs:
        assert meta.loc.fs.exists(art.uri)
    else:
        assert isinstance(art, LocalArtifact)
        assert art.uri == filename
    loaded_meta = load_meta(out_path, load_value=True)
    loaded = loaded_meta.get_value()
    loaded.predict(train)


@pytest.mark.parametrize("file_ext, type_", [(".pkl", None), ("", "pickle")])
def test_import_model_pickle__no_copy(
    write_model_pickle, train, tmpdir, file_ext, type_
):
    path = str(tmpdir / IMPORT_MODEL_FILENAME + file_ext)
    write_model_pickle(path)

    out_path = str(tmpdir / "mlem_model")
    meta = import_object(path, target=out_path, type_=type_, copy_data=False)
    _check_meta(meta, out_path)
    _check_load_artifact(meta, out_path, True, train)


@pytest.mark.parametrize("file_ext, type_", [(".pkl", None), ("", "pickle")])
def test_import_model_pickle__no_copy_in_mlem_repo(
    write_model_pickle, train, mlem_repo, file_ext, type_
):
    filename = IMPORT_MODEL_FILENAME + file_ext
    path = os.path.join(mlem_repo, filename)
    write_model_pickle(path)

    out_path = os.path.join(mlem_repo, "mlem_model")
    meta = import_object(
        path, target=out_path, type_=type_, copy_data=False, external=True
    )
    _check_meta(meta, out_path)
    _check_load_artifact(meta, out_path, False, train, filename)


@long
def test_import_model_pickle_remote(
    s3_tmp_path, s3_storage_fs, write_model_pickle, tmpdir, train
):
    path = posixpath.join(
        s3_tmp_path("import_model_no_repo"), IMPORT_MODEL_FILENAME
    )
    write_model_pickle(path, s3_storage_fs)
    out_path = str(tmpdir / "mlem_model")
    meta = import_object(
        path, target=out_path, copy_data=False, type_="pickle"
    )
    _check_meta(meta, out_path)

    loaded = load(out_path)
    loaded.predict(train)


@long
def test_import_model_pickle_remote_in_repo(
    s3_tmp_path, s3_storage_fs, write_model_pickle, train
):
    repo_path = s3_tmp_path("import_model_repo")
    init(repo_path)
    path = posixpath.join(repo_path, IMPORT_MODEL_FILENAME)
    write_model_pickle(path, s3_storage_fs)
    out_path = posixpath.join(repo_path, "mlem_model")
    meta = import_object(
        path, target=out_path, copy_data=False, type_="pickle", external=True
    )
    _check_meta(meta, out_path, s3_storage_fs)
    _check_load_artifact(meta, out_path, False, train)
