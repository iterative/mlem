import os
import tempfile

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

from mlem.core.meta_io import META_FILE_NAME, MLEM_DIR, MLEM_EXT
from mlem.core.metadata import load, load_meta
from mlem.core.objects import MlemLink, ModelMeta, mlem_dir_path


def test_model_dump(mlem_root):
    X, y = load_iris(return_X_y=True)
    clf = DecisionTreeClassifier().fit(X, y)
    meta = ModelMeta.from_obj(clf, test_data=X)
    dir = os.path.join(mlem_root, "decision_tree")
    meta.dump(dir, link=True)
    link_path = os.path.join(
        mlem_root, MLEM_DIR, "model", "decision_tree" + MLEM_EXT
    )
    assert os.path.exists(link_path)
    model = load(link_path, follow_links=True)
    model.predict(X)


def test_model_cloning(model_path):
    model = load_meta(model_path)
    with tempfile.TemporaryDirectory() as dir:
        model.clone(dir, link=False)
        cloned_model = load(dir)
        X, y = load_iris(return_X_y=True)
        cloned_model.predict(X)


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
