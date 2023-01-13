from pathlib import PosixPath

import pytest

from mlem.api.utils import get_model_meta
from mlem.core.objects import MlemModel, ModelAnalyzer


@pytest.fixture
def custom_processors_model(tmp_path: PosixPath):
    def funclen(x):
        return len(x)

    model = MlemModel()
    model.add_processor(
        "explain", ModelAnalyzer.analyze(funclen, sample_data="word")
    )
    model.add_processor(
        "guess", ModelAnalyzer.analyze(funclen, sample_data="word")
    )
    model.call_orders["explain"] = [("explain", "__call__")]
    model.call_orders["guess"] = [("guess", "__call__")]
    path = str(tmp_path / "model")
    model.dump(path)
    return path


def test_get_model_meta(custom_processors_model):
    get_model_meta(custom_processors_model, load_value=True)
