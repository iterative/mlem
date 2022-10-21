import lightgbm as lgb
import numpy as np
import pytest
from pydantic.error_wrappers import ValidationError

from mlem.contrib.requirements import RequirementsBuilder
from mlem.core.objects import MlemModel


def test_build_reqs(tmp_path, model_meta):
    path = str(tmp_path / "reqs.txt")
    builder = RequirementsBuilder(target=path)
    builder.build(model_meta)
    with open(path, "r", encoding="utf-8") as f:
        assert model_meta.requirements.to_pip() == f.read().splitlines()


def test_build_reqs_with_invalid_req_type():
    with pytest.raises(
        ValidationError, match="req_type invalid is not valid."
    ):
        RequirementsBuilder(req_type="invalid")


def test_build_requirements_should_print_with_no_path(capsys, model_meta):
    builder = RequirementsBuilder()
    builder.build(model_meta)
    captured = capsys.readouterr()
    assert captured.out == " ".join(model_meta.requirements.to_pip()) + "\n"


def test_unix_requirement(capsys):
    np_payload = np.linspace(0, 2, 5).reshape((-1, 1))
    data_np = lgb.Dataset(
        np_payload,
        label=np_payload.reshape((-1,)).tolist(),
        free_raw_data=False,
    )
    booster = lgb.train({}, data_np, 1)
    model = MlemModel.from_obj(booster, sample_data=data_np)
    builder = RequirementsBuilder(req_type="unix")
    builder.build(model)
    captured = capsys.readouterr()
    assert str(captured.out).endswith(
        "\n".join(model.requirements.to_unix()) + "\n"
    )
