import pytest

from mlem.contrib.venv import VenvBuilder
from mlem.core.errors import MlemError


def test_build_venv(tmp_path, model_meta):
    path = str(tmp_path / "venv")
    builder = VenvBuilder(target=path)
    env_exe = builder.build(model_meta)
    installed_pkgs = (
        builder.get_installed_packages(env_exe).decode().splitlines()
    )
    for each_req in model_meta.requirements.to_pip():
        assert each_req in installed_pkgs


def test_install_in_current_venv_not_active(tmp_path, model_meta):
    path = str(tmp_path / "venv")
    builder = VenvBuilder(target=path, current_env=True)
    with pytest.raises(MlemError) as e:
        builder.build(model_meta)
        assert "No virtual environment detected" in str(e.value)
