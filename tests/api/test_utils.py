from pathlib import PosixPath

import pytest

from mlem.api.utils import get_model_meta
from mlem.core.objects import MlemModel, ModelAnalyzer


def funclen(x):
    return len(x)


@pytest.fixture
def one_custom_processor_model(tmp_path: PosixPath):
    model = MlemModel()
    model.add_processor(
        "textlen", ModelAnalyzer.analyze(funclen, sample_data="word")
    )
    model.call_orders["textlen"] = [("textlen", "__call__")]
    path = str(tmp_path / "model")
    model.dump(path)
    return path


@pytest.fixture
def two_custom_processors_model(tmp_path: PosixPath):
    model = MlemModel()
    model.add_processor(
        "textlen", ModelAnalyzer.analyze(funclen, sample_data="word")
    )
    model.call_orders["textlen"] = [("textlen", "__call__")]
    model.add_processor(
        "textlen2", ModelAnalyzer.analyze(funclen, sample_data="word")
    )
    model.call_orders["textlen2"] = [("textlen2", "__call__")]
    path = str(tmp_path / "model")
    model.dump(path)
    return path


def test_get_model_meta_one_processor(one_custom_processor_model):
    model = get_model_meta(one_custom_processor_model, load_value=True)
    assert model.textlen("tenletters") == 10


def test_get_model_meta_two_processors(two_custom_processors_model):
    model = get_model_meta(two_custom_processors_model, load_value=True)
    assert model.textlen("tenletters") == 10
