import os
import tempfile

import pytest
import yaml
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

from mlem.api import init
from mlem.core.metadata import load, load_meta, save
from tests.conftest import long


def test_model_saving_without_sample_data(model, tmpdir_factory):
    dir = str(tmpdir_factory.mktemp("saving-models-without-sample-data"))
    # link=True would require having .mlem folder somewhere
    save(model, dir, link=False)


def test_model_saving_in_mlem_root(model_train_target, tmpdir_factory):
    mlem_root = str(tmpdir_factory.mktemp("mlem-root"))
    init(mlem_root)
    model_dir = os.path.join(mlem_root, "generated-model")
    model, train, _ = model_train_target
    save(model, model_dir, tmp_sample_data=train, link=True)


def test_model_loading(model_path):
    model = load(model_path)
    assert isinstance(model, DecisionTreeClassifier)
    train, _ = load_iris(return_X_y=True)
    model.predict(train)


def test_meta_loading(model_path):
    model = load_meta(model_path, load_value=True)
    assert isinstance(model.model.model, DecisionTreeClassifier)
    train, _ = load_iris(return_X_y=True)
    model.model.model.predict(train)


@long
@pytest.mark.parametrize(
    "url",
    [
        "github://iterative:example-mlem@main/data/model",
        "github://iterative:example-mlem@main/data/model/mlem.yaml",
        "github://iterative:example-mlem@main/.mlem/model/data/model.mlem.yaml",
        "github://iterative:example-mlem@main/.mlem/model/latest.mlem.yaml",
    ],
)
def test_model_loading_from_github_with_fsspec(url):
    assert "GITHUB_USERNAME" in os.environ and "GITHUB_TOKEN" in os.environ
    model = load(url)
    train, _ = load_iris(return_X_y=True)
    model.predict(train)


@long
@pytest.mark.parametrize(
    "path",
    [
        "data/model",
        "data/model/mlem.yaml",
        ".mlem/model/data/model.mlem.yaml",
        ".mlem/model/latest.mlem.yaml",
    ],
)
def test_model_loading_from_github(path):
    assert "GITHUB_USERNAME" in os.environ and "GITHUB_TOKEN" in os.environ
    model = load(
        path,
        repo="https://github.com/iterative/example-mlem",
        rev="main",
    )
    train, _ = load_iris(return_X_y=True)
    model.predict(train)


def test_load_link_with_fsspec_path():
    link_contents = {
        "link_type": "model",
        "mlem_link": "github://iterative:example-mlem@main/data/model/mlem.yaml",
        "object_type": "link",
    }
    with tempfile.TemporaryDirectory() as dir:
        path = os.path.join(dir, "link.mlem.yaml")
        with open(path, "w", encoding="utf-8") as f:
            f.write(yaml.safe_dump(link_contents))
        model = load(path)
        train, _ = load_iris(return_X_y=True)
        model.predict(train)
