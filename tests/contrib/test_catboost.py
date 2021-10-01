import numpy as np
import pytest
from catboost import CatBoostClassifier, CatBoostRegressor
from fsspec.implementations.local import LocalFileSystem

from mlem.core.model import ModelAnalyzer


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
    "catboost_model", ["catboost_classifier", "catboost_regressor"]
)
def test_catboost_model(catboost_model, pandas_data, tmpdir, request):
    catboost_model = request.getfixturevalue(catboost_model)

    cbmw = ModelAnalyzer.analyze(catboost_model, test_data=pandas_data)
    expected_requirements = {
        "catboost",
    }  # 'pandas', 'numpy'} # TODO: https://github.com/iterative/mlem/issues/21 methods
    assert set(cbmw.get_requirements().modules) == expected_requirements
    assert cbmw.model is catboost_model

    cbmw.dump(LocalFileSystem(), tmpdir)

    cbmw.model = None
    with pytest.raises(ValueError):
        cbmw.call_method("predict", pandas_data)

    cbmw.load(LocalFileSystem(), tmpdir)
    assert cbmw.model is not catboost_model
    assert set(cbmw.get_requirements().modules) == expected_requirements

    np.testing.assert_array_almost_equal(
        catboost_model.predict(pandas_data),
        cbmw.call_method("predict", pandas_data),
    )

    if isinstance(catboost_model, CatBoostClassifier):
        np.testing.assert_array_almost_equal(
            catboost_model.predict_proba(pandas_data),
            cbmw.call_method("predict_proba", pandas_data),
        )
    else:
        assert "predict_proba" not in cbmw.methods
