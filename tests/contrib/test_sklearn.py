import posixpath

import lightgbm as lgb
import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from mlem.contrib.numpy import NumpyNdarrayType
from mlem.contrib.sklearn import SklearnModel
from mlem.core.artifacts import LOCAL_STORAGE
from mlem.core.dataset_type import DatasetAnalyzer
from mlem.core.model import ModelAnalyzer
from mlem.core.requirements import UnixPackageRequirement
from tests.conftest import check_model_type_common_interface, long


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


@pytest.fixture
def lgbm_model(inp_data, out_data):
    lgbm_regressor = lgb.LGBMRegressor()
    return lgbm_regressor.fit(inp_data, out_data)


@pytest.mark.parametrize("model_fixture", ["classifier", "regressor"])
def test_hook(model_fixture, inp_data, request):
    model = request.getfixturevalue(model_fixture)
    data_type = DatasetAnalyzer.analyze(inp_data)
    model_type = ModelAnalyzer.analyze(model, sample_data=inp_data)

    assert isinstance(model_type, SklearnModel)
    check_model_type_common_interface(
        model_type,
        data_type,
        NumpyNdarrayType(
            shape=(None,),
            dtype=model.predict(inp_data).dtype.name,
        ),
    )


def test_hook_lgb(lgbm_model, inp_data):
    data_type = DatasetAnalyzer.analyze(inp_data)
    model_type = ModelAnalyzer.analyze(lgbm_model, sample_data=inp_data)

    assert isinstance(model_type, SklearnModel)
    check_model_type_common_interface(
        model_type,
        data_type,
        NumpyNdarrayType(
            shape=(None,),
            dtype="float64",
        ),
        varkw="kwargs",
    )


@pytest.mark.parametrize("model", ["classifier", "regressor"])
def test_model_type__predict(model, inp_data, request):
    model = request.getfixturevalue(model)
    model_type = ModelAnalyzer.analyze(model, sample_data=inp_data)

    np.testing.assert_array_almost_equal(
        model.predict(inp_data), model_type.call_method("predict", inp_data)
    )


def test_model_type__clf_predict_proba(classifier, inp_data):
    model_type = ModelAnalyzer.analyze(classifier, sample_data=inp_data)

    np.testing.assert_array_almost_equal(
        classifier.predict_proba(inp_data),
        model_type.call_method("predict_proba", inp_data),
    )


def test_model_type__reg_predict_proba(regressor, inp_data):
    model_type = ModelAnalyzer.analyze(regressor, sample_data=inp_data)

    with pytest.raises(ValueError):
        model_type.call_method("predict_proba", inp_data)


@pytest.mark.parametrize("model", ["classifier", "regressor"])
def test_model_type__dump_load(tmpdir, model, inp_data, request):
    model = request.getfixturevalue(model)
    model_type = ModelAnalyzer.analyze(model, sample_data=inp_data)

    expected_requirements = {"sklearn", "numpy"}
    assert set(model_type.get_requirements().modules) == expected_requirements
    artifacts = model_type.dump(LOCAL_STORAGE, posixpath.join(tmpdir, "model"))
    model_type.model = None

    with pytest.raises(ValueError):
        model_type.call_method("predict", inp_data)

    model_type.load(artifacts)
    np.testing.assert_array_almost_equal(
        model.predict(inp_data), model_type.call_method("predict", inp_data)
    )
    assert set(model_type.get_requirements().modules) == expected_requirements


@long
def test_model_type_lgb__dump_load(tmpdir, lgbm_model, inp_data):
    model_type = ModelAnalyzer.analyze(lgbm_model, sample_data=inp_data)

    expected_requirements = {"sklearn", "lightgbm", "pandas", "numpy", "scipy"}
    reqs = model_type.get_requirements().expanded
    assert set(reqs.modules) == expected_requirements
    assert reqs.of_type(UnixPackageRequirement) == [
        UnixPackageRequirement(package_name="libgomp1")
    ]
    artifacts = model_type.dump(LOCAL_STORAGE, str(tmpdir / "model"))
    model_type.model = None

    with pytest.raises(ValueError):
        model_type.call_method("predict", inp_data)

    model_type.load(artifacts)
    np.testing.assert_array_almost_equal(
        lgbm_model.predict(inp_data),
        model_type.call_method("predict", inp_data),
    )
    reqs = model_type.get_requirements().expanded
    assert set(reqs.modules) == expected_requirements
    assert reqs.of_type(UnixPackageRequirement) == [
        UnixPackageRequirement(package_name="libgomp1")
    ]


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
