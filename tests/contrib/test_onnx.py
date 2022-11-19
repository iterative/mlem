import posixpath

import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import Int64TensorType
from sklearn.linear_model import LinearRegression, LogisticRegression

from mlem.contrib.numpy import NumpyNdarrayType
from mlem.contrib.onnx import ONNXModel
from mlem.core.artifacts import LOCAL_STORAGE
from mlem.core.data_type import DataAnalyzer, ListType
from mlem.core.model import Argument, ModelAnalyzer


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
    initial_type = [("int_input", Int64TensorType([None, 3]))]
    lr_onnx = convert_sklearn(lr, initial_types=initial_type)
    return lr_onnx


@pytest.fixture
def onnx_regressor(inp_data_nparray, out_data_nparray):
    lr = LinearRegression()
    lr.fit(inp_data_nparray, out_data_nparray)
    initial_type = [("int_input", Int64TensorType([None, 3]))]
    lr_onnx = convert_sklearn(lr, initial_types=initial_type)
    return lr_onnx


@pytest.fixture
def onnx_classifier_inference():
    return [
        np.array([1, 2]),
        np.array([[0.73935163, 0.26064837], [0.26064837, 0.73935163]]),
    ]


@pytest.fixture
def onnx_regressor_inference():
    return [np.array([[1.0], [2.0]])]


test_hook_classifier_data = [
    (
        lazy_fixture("onnx_classifier"),
        [
            NumpyNdarrayType(data=None, shape=(None,), dtype="int64"),
            NumpyNdarrayType(data=None, shape=(None, 2), dtype="float64"),
        ],
    ),
    (
        lazy_fixture("onnx_regressor"),
        [NumpyNdarrayType(data=None, shape=(None, 1), dtype="float32")],
    ),
]


@pytest.mark.parametrize(
    "model,expected_return_type_items", test_hook_classifier_data
)
def test_hook_classifier(model, expected_return_type_items, inp_data_nparray):
    data_type = DataAnalyzer.analyze(inp_data_nparray)
    model_type = ModelAnalyzer.analyze(model, sample_data=inp_data_nparray)

    assert isinstance(model_type, ONNXModel)
    returns = ListType(items=expected_return_type_items)
    signature = model_type.methods["predict"]
    assert signature.name == "predict"
    assert signature.args[0] == Argument(name="data", type_=data_type)
    assert signature.returns == returns


@pytest.mark.parametrize(
    "model,expected_output,output_idx",
    [
        (
            lazy_fixture("onnx_classifier"),
            lazy_fixture("onnx_classifier_inference"),
            0,
        ),
        (
            lazy_fixture("onnx_classifier"),
            lazy_fixture("onnx_classifier_inference"),
            1,
        ),
        (
            lazy_fixture("onnx_regressor"),
            lazy_fixture("onnx_regressor_inference"),
            0,
        ),
    ],
)
def test_model_type__predict(
    model, expected_output, output_idx, inp_data_nparray
):
    model_type = ModelAnalyzer.analyze(model, sample_data=inp_data_nparray)

    np.testing.assert_array_almost_equal(
        expected_output[output_idx],
        model_type.call_method("predict", inp_data_nparray)[output_idx],
    )


def test_model_type__clf_missing_method(onnx_classifier, inp_data_nparray):
    model_type = ModelAnalyzer.analyze(
        onnx_classifier, sample_data=inp_data_nparray
    )

    with pytest.raises(
        ValueError, match=r"doesn't expose method 'predict_proba'"
    ):
        model_type.call_method("predict_proba", inp_data_nparray)


def test_model_type__reg_predict_incorrect_inp_data(
    onnx_classifier, inp_data_nparray
):
    model_type = ModelAnalyzer.analyze(
        onnx_classifier, sample_data=inp_data_nparray
    )

    with pytest.raises(
        ValueError, match="no of inputs provided: 2, expected: 1"
    ):
        model_type.call_method("predict", [inp_data_nparray, inp_data_nparray])


@pytest.mark.parametrize(
    "model,expected_output,output_idx",
    [
        (
            lazy_fixture("onnx_classifier"),
            lazy_fixture("onnx_classifier_inference"),
            0,
        ),
        (
            lazy_fixture("onnx_classifier"),
            lazy_fixture("onnx_classifier_inference"),
            1,
        ),
        (
            lazy_fixture("onnx_regressor"),
            lazy_fixture("onnx_regressor_inference"),
            0,
        ),
    ],
)
def test_model_type__dump_load(
    tmpdir, model, expected_output, output_idx, inp_data_nparray
):
    model_type = ModelAnalyzer.analyze(model, sample_data=inp_data_nparray)

    expected_requirements = {
        "numpy",
        "onnx",
        "pandas",
        "onnxruntime",
        "protobuf",
    }
    assert set(model_type.get_requirements().modules) == expected_requirements
    artifacts = model_type.dump(LOCAL_STORAGE, posixpath.join(tmpdir, "model"))
    model_type.model = None

    with pytest.raises(ValueError):
        model_type.call_method("predict", inp_data_nparray)

    model_type.load(artifacts)
    np.testing.assert_array_almost_equal(
        expected_output[output_idx],
        model_type.call_method("predict", inp_data_nparray)[output_idx],
    )
    assert set(model_type.get_requirements().modules) == expected_requirements
