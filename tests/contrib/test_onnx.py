import posixpath

import numpy as np
import pandas as pd
import pytest
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import Int64TensorType
from sklearn.linear_model import LogisticRegression, LinearRegression

from mlem.contrib.numpy import NumpyNdarrayType
from mlem.contrib.onnx import ONNXModel, ONNXWrappedModel
from mlem.core.artifacts import LOCAL_STORAGE
from mlem.core.data_type import DataAnalyzer, ListType
from mlem.core.model import ModelAnalyzer
from tests.conftest import check_model_type_common_interface


@pytest.fixture
def inp_data_nparray():
    return np.array([[1, 2, 3], [3, 2, 1]])


@pytest.fixture
def out_data_nparray():
    return np.array([1, 2])


@pytest.fixture
def onnx_classifier(inp_data_nparray, out_data_nparray):
    lr = LogisticRegression()
    lr.fit(inp_data_nparray, out_data_nparray)
    initial_type = [('int_input', Int64TensorType([None, 3]))]
    lr_onnx = convert_sklearn(lr, initial_types=initial_type)
    return lr_onnx


@pytest.fixture
def onnx_regressor(inp_data_nparray, out_data_nparray):
    lr = LinearRegression()
    lr.fit(inp_data_nparray, out_data_nparray)
    initial_type = [('int_input', Int64TensorType([None, 3]))]
    lr_onnx = convert_sklearn(lr, initial_types=initial_type)
    return lr_onnx


def test_hook_classifier(onnx_classifier, inp_data_nparray):
    data_type = DataAnalyzer.analyze(inp_data_nparray)
    model_type = ModelAnalyzer.analyze(onnx_classifier, sample_data=inp_data_nparray)

    assert isinstance(model_type, ONNXModel)
    check_model_type_common_interface(
        model_type,
        data_type,
        ListType(items=[NumpyNdarrayType(data=None, shape=(None,), dtype='int64'),
                        NumpyNdarrayType(data=None, shape=(None, 2), dtype='float64')])
    )


def test_hook_regressor(onnx_regressor, inp_data_nparray):
    data_type = DataAnalyzer.analyze(inp_data_nparray)
    model_type = ModelAnalyzer.analyze(onnx_regressor, sample_data=inp_data_nparray)

    assert isinstance(model_type, ONNXModel)
    check_model_type_common_interface(
        model_type,
        data_type,
        ListType(items=[NumpyNdarrayType(data=None, shape=(None, 1), dtype='float32')])
    )


@pytest.mark.parametrize("model", ["onnx_classifier", "onnx_regressor"])
def test_model_type__predict(model, inp_data_nparray, request):
    model = request.getfixturevalue(model)
    model_type = ModelAnalyzer.analyze(model, sample_data=inp_data_nparray)

    np.testing.assert_array_almost_equal(
        ONNXWrappedModel(model).predict(inp_data_nparray)[0], model_type.call_method("predict", inp_data_nparray)[0]
    )


def test_model_type__clf_predict_proba(onnx_classifier, inp_data_nparray):
    model_type = ModelAnalyzer.analyze(onnx_classifier, sample_data=inp_data_nparray)

    pd.testing.assert_frame_equal(
        pd.DataFrame(ONNXWrappedModel(onnx_classifier).predict(inp_data_nparray)[1]),
        pd.DataFrame(model_type.call_method("predict", inp_data_nparray)[1])
    )


def test_model_type__reg_predict_proba(onnx_classifier, inp_data_nparray):
    model_type = ModelAnalyzer.analyze(onnx_classifier, sample_data=inp_data_nparray)

    with pytest.raises(ValueError):
        model_type.call_method("predict_proba", inp_data_nparray)


def test_model_type__reg_predict_incorrect_inp_data(onnx_classifier, inp_data_nparray):
    model_type = ModelAnalyzer.analyze(onnx_classifier, sample_data=inp_data_nparray)

    with pytest.raises(ValueError):
        model_type.call_method("predict", [inp_data_nparray, inp_data_nparray])


@pytest.mark.parametrize("model", ["onnx_classifier", "onnx_regressor"])
def test_model_type__dump_load(tmpdir, model, inp_data_nparray, request):
    model = request.getfixturevalue(model)
    model_type = ModelAnalyzer.analyze(model, sample_data=inp_data_nparray)

    expected_requirements = {'numpy', 'onnx', 'pandas', 'onnxruntime'}
    assert set(model_type.get_requirements().modules) == expected_requirements
    artifacts = model_type.dump(LOCAL_STORAGE, posixpath.join(tmpdir, "model"))
    model_type.model = None

    with pytest.raises(ValueError):
        model_type.call_method("predict", inp_data_nparray)

    model_type.load(artifacts)
    np.testing.assert_array_almost_equal(
        ONNXWrappedModel(model).predict(inp_data_nparray)[0], model_type.call_method("predict", inp_data_nparray)[0]
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
