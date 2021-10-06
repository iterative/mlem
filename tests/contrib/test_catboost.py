import numpy as np
import pytest
from catboost import CatBoostClassifier, CatBoostRegressor
from fsspec.implementations.local import LocalFileSystem

from mlem.constants import PREDICT_METHOD_NAME, PREDICT_PROBA_METHOD_NAME
from mlem.contrib.numpy import NumpyNdarrayType
from mlem.core.dataset_type import DatasetAnalyzer
from mlem.core.model import ModelAnalyzer
from tests.conftest import check_model_type_common_interface


@pytest.fixture
def catboost_params(tmpdir):
    return {"iterations": 1, "train_dir": str(tmpdir)}


@pytest.fixture
def catboost_classifier(pandas_data, catboost_params):
    return CatBoostClassifier(**catboost_params).fit(pandas_data, [1, 0])


@pytest.fixture
def catboost_regressor(pandas_data, catboost_params):
    return CatBoostRegressor(**catboost_params).fit(pandas_data, [1, 0])


@pytest.mark.parametrize(
    "catboost_model_fixture", ["catboost_classifier", "catboost_regressor"]
)
def test_catboost_model(catboost_model_fixture, pandas_data, tmpdir, request):
    catboost_model = request.getfixturevalue(catboost_model_fixture)

    cbmw = ModelAnalyzer.analyze(catboost_model, sample_data=pandas_data)

    data_type = DatasetAnalyzer.analyze(pandas_data)

    assert "catboost_predict" in cbmw.methods
    check_model_type_common_interface(
        cbmw,
        data_type,
        NumpyNdarrayType(
            shape=(None,),
            dtype="float64"
            if catboost_model_fixture == "catboost_regressor"
            else "int64",
        ),
    )

    expected_requirements = {"catboost", "pandas", "numpy"}
    assert set(cbmw.get_requirements().modules) == expected_requirements
    assert cbmw.model is catboost_model

    cbmw.dump(LocalFileSystem(), tmpdir)

    cbmw.model = None
    with pytest.raises(ValueError):
        cbmw.call_method(PREDICT_METHOD_NAME, pandas_data)

    cbmw.load(LocalFileSystem(), tmpdir)
    assert cbmw.model is not catboost_model
    assert set(cbmw.get_requirements().modules) == expected_requirements

    np.testing.assert_array_almost_equal(
        catboost_model.predict(pandas_data),
        cbmw.call_method(PREDICT_METHOD_NAME, pandas_data),
    )

    if isinstance(catboost_model, CatBoostClassifier):
        np.testing.assert_array_almost_equal(
            catboost_model.predict_proba(pandas_data),
            cbmw.call_method(PREDICT_PROBA_METHOD_NAME, pandas_data),
        )
    else:
        assert PREDICT_PROBA_METHOD_NAME not in cbmw.methods
