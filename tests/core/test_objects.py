import os
import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

from mlem.core.meta_io import (
    ART_DIR,
    META_FILE_NAME,
    MLEM_DIR,
    MLEM_EXT,
    deserialize,
)
from mlem.core.metadata import load, load_meta
from mlem.core.objects import MlemLink, ModelMeta, mlem_dir_path
from tests.conftest import MLEM_TEST_REPO, long, need_test_repo_auth

MODEL_NAME = "decision_tree"


def test_model_dump(mlem_root):
    X, y = load_iris(return_X_y=True)
    clf = DecisionTreeClassifier().fit(X, y)
    meta = ModelMeta.from_obj(clf, sample_data=X)
    dir = os.path.join(mlem_root, MODEL_NAME)
    meta.dump(dir, link=True)
    link_path = os.path.join(
        mlem_root, MLEM_DIR, "model", MODEL_NAME + MLEM_EXT
    )
    assert os.path.exists(link_path)
    assert os.path.exists(os.path.join(dir, META_FILE_NAME))
    assert os.path.exists(
        os.path.join(mlem_root, MODEL_NAME, ART_DIR, "data.pkl")
    )
    model = load(link_path, follow_links=True)
    model.predict(X)


def test_model_cloning(model_path):
    model = load_meta(model_path)
    with tempfile.TemporaryDirectory() as dir:
        model.clone(dir, link=False)
        cloned_model = load(dir)
        X, _ = load_iris(return_X_y=True)
        cloned_model.predict(X)


@long
@need_test_repo_auth
def test_model_cloning_remote(current_test_branch):
    """TODO: https://github.com/iterative/mlem/issues/44
    test fails in CI because repo is private and DVC does not support http auth for git
    """
    with tempfile.TemporaryDirectory() as dir:
        cloned_model = load_meta(
            os.path.join(MLEM_TEST_REPO, "simple/data/model"),
            rev=current_test_branch,
        ).clone(os.path.join(dir, "model"), link=False)
        cloned_model.load_value()
        X, _ = load_iris(return_X_y=True)
        cloned_model.predict(X)


def test_model_getattr(model_meta):
    method = model_meta.predict
    assert callable(method)
    X, _ = load_iris(return_X_y=True)
    method(X)

    with pytest.raises(AttributeError):
        model_meta.not_existing_method(X)


def test_mlem_dir_path(mlem_root):
    # case when we provide objects' abspath and object is already located in the same MLEM root
    model_link = os.path.join(
        mlem_root, MLEM_DIR, "model", "data", "model" + MLEM_EXT
    )
    assert (
        mlem_dir_path(
            os.path.join(mlem_root, "data", "model"), obj_type="model", fs=None
        )
        == model_link
    )
    # case when we provide object relative path
    model_link = os.path.join(
        mlem_root, MLEM_DIR, "model", "latest" + MLEM_EXT
    )
    assert (
        mlem_dir_path("latest", fs=None, obj_type="model", mlem_root=mlem_root)
        == model_link
    )


def test_link_dump(model_path):
    link = MlemLink(
        mlem_link=os.path.join(model_path, META_FILE_NAME), link_type="model"
    )
    with tempfile.TemporaryDirectory() as dir:
        path_to_link = os.path.join(dir, "latest" + MLEM_EXT)
        link.dump(path_to_link, absolute=True)
        model = load_meta(path_to_link, follow_links=True)
    assert isinstance(model, ModelMeta)


def test_link_dump_in_mlem(model_path_mlem_root):
    model_path, mlem_root = model_path_mlem_root
    link = MlemLink(
        mlem_link=os.path.join(model_path, META_FILE_NAME), link_type="model"
    )
    link_name = "latest"
    link.dump(link_name, mlem_root=mlem_root)
    model = load_meta(os.path.join(mlem_root, link_name), follow_links=True)
    assert isinstance(model, ModelMeta)


def test_model_model_type_laziness():
    payload = {
        "model_type": {"type": "doesnotexist"},
        "object_type": "model",
        "requirements": [],
    }
    model = deserialize(payload, ModelMeta)
    assert model.model_type_raw == {"type": "doesnotexist"}
    with pytest.raises(ValidationError):
        print(model.model_type)


def test_mlem_root(mlem_root):
    path = Path(mlem_root)
    assert os.path.exists(path)
    assert os.path.isdir(path)
    mlem_dir = path / MLEM_DIR
    assert os.path.isdir(mlem_dir)
    assert os.path.isfile(mlem_dir / "model" / ("model1" + MLEM_EXT))
    assert os.path.isfile(mlem_dir / "model" / ("latest" + MLEM_EXT))
    model_dir = path / "model1"
    assert os.path.isdir(model_dir)
    assert os.path.isfile(model_dir / META_FILE_NAME)
    assert os.path.isfile(model_dir / ART_DIR / "data.pkl")
