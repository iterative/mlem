import os
import re

import pytest

from mlem.contrib.venv import CondaBuilder, VenvBuilder
from mlem.core.errors import MlemError
from mlem.core.requirements import (
    CondaPackageRequirement,
    InstallableRequirement,
)
from tests.contrib.conftest import conda_test


def process_conda_list_output(installed_pkgs):
    def get_words(line):
        return re.findall(r"[^\s]+", line)

    words = [get_words(x) for x in installed_pkgs.splitlines()[3:]]
    keys = []
    vals = []
    for w in words:
        if len(w) >= 4:
            keys.append(w[0])
            vals.append(w[3])
    result = dict(zip(keys, vals))
    return result


@conda_test
def test_build_conda(tmp_path, model_meta):
    path = str(tmp_path / "venv")
    builder = CondaBuilder(target=path)
    model_meta.requirements.add(
        CondaPackageRequirement(package_name="xtensor")
    )
    env_dir = builder.build(model_meta)
    installed_pkgs = builder.get_installed_packages(env_dir).decode()
    pkgs_info = process_conda_list_output(installed_pkgs)
    for each_req in model_meta.requirements:
        if isinstance(each_req, InstallableRequirement):
            assert pkgs_info[each_req.package] == "pypi"
        elif isinstance(each_req, CondaPackageRequirement):
            assert pkgs_info[each_req.package_name] == each_req.channel_name


def test_build_venv(tmp_path, model_meta):
    path = str(tmp_path / "venv")
    builder = VenvBuilder(target=path)
    env_dir = builder.build(model_meta)
    installed_pkgs = (
        builder.get_installed_packages(env_dir).decode().splitlines()
    )
    for each_req in model_meta.requirements.to_pip():
        assert each_req in installed_pkgs


def test_install_in_current_venv_not_active(tmp_path, model_meta):
    path = str(tmp_path / "venv")
    builder = VenvBuilder(target=path, current_env=True)
    with pytest.raises(MlemError) as e:
        builder.build(model_meta)
    assert "No virtual environment detected" in str(e.value)


def test_install_in_current_active_venv(tmp_path, model_meta):
    path = str(tmp_path / "venv-act")
    builder = VenvBuilder(target=path)
    builder.create_virtual_env()
    assert (
        builder.get_installed_packages(builder.context.env_dir).decode() == ""
    )
    os.environ["VIRTUAL_ENV"] = builder.context.env_dir
    builder.current_env = True
    builder.build(model_meta)
    installed_pkgs = (
        builder.get_installed_packages(builder.context.env_dir)
        .decode()
        .splitlines()
    )
    for each_req in model_meta.requirements.to_pip():
        assert each_req in installed_pkgs
