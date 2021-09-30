import numpy as np
import pytest
from fsspec.implementations.local import LocalFileSystem
from sklearn.linear_model import LinearRegression, LogisticRegression

from mlem.contrib.sklearn import SklearnModel
from mlem.core.model import ModelAnalyzer


@pytest.fixture
def inp_data():
    return [[1, 2, 3], [3, 2, 1]]


@pytest.fixture
def out_data():
    return [1, 2]


@pytest.fixture
def classifier(inp_data, out_data):
    lr = LogisticRegression()
    lr.fit(inp_data, out_data)
    return lr


@pytest.fixture
def regressor(inp_data, out_data):
    lr = LinearRegression()
    lr.fit(inp_data, out_data)
    return lr


@pytest.mark.parametrize("model", ["classifier", "regressor"])
def test_hook(model, inp_data, request):
    model = request.getfixturevalue(model)
    model_type = ModelAnalyzer.analyze(model, test_data=inp_data)

    assert isinstance(model_type, SklearnModel)


@pytest.mark.parametrize("model", ["classifier", "regressor"])
def test_model_type__predict(model, inp_data, request):
    model = request.getfixturevalue(model)
    model_type = ModelAnalyzer.analyze(model, test_data=inp_data)

    np.testing.assert_array_almost_equal(
        model.predict(inp_data), model_type.call_method("predict", inp_data)
    )


def test_model_type__clf_predict_proba(classifier, inp_data):
    model_type = ModelAnalyzer.analyze(classifier, test_data=inp_data)

    np.testing.assert_array_almost_equal(
        classifier.predict_proba(inp_data),
        model_type.call_method("predict_proba", inp_data),
    )


def test_model_type__reg_predict_proba(regressor, inp_data):
    model_type = ModelAnalyzer.analyze(regressor, test_data=inp_data)

    with pytest.raises(ValueError):
        model_type.call_method("predict_proba", inp_data)


@pytest.mark.parametrize("model", ["classifier", "regressor"])
def test_model_type__dump_load(tmpdir, model, inp_data, request):
    model = request.getfixturevalue(model)
    model_type = ModelAnalyzer.analyze(model, test_data=inp_data)

    expected_requirements = {"sklearn", "numpy"}
    assert set(model_type.get_requirements().modules) == expected_requirements
    model_type.dump(LocalFileSystem(), tmpdir)
    model_type.model = None

    with pytest.raises(ValueError):
        model_type.call_method("predict", inp_data)

    model_type.load(LocalFileSystem(), tmpdir)
    np.testing.assert_array_almost_equal(
        model.predict(inp_data), model_type.call_method("predict", inp_data)
    )
    assert set(model_type.get_requirements().modules) == expected_requirements


# Copyright 2019 Zyfra
# Copyright 2021 Iterative
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
