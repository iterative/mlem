import os
import subprocess

from mlem.contrib.pip.base import PipPackager, WhlPackager

PIP_PACKAGE_NAME = "test_pip_package_name"


def test_pip_package(tmpdir, model_meta_saved_single):
    path = str(tmpdir)
    packager = PipPackager(target=path, package_name=PIP_PACKAGE_NAME)
    packager.package(model_meta_saved_single)

    print(
        subprocess.check_output(
            "pip install -e . --no-deps", shell=True, cwd=path
        ).decode("utf8")
    )
    try:
        subprocess.check_output(
            f'python -c "import {PIP_PACKAGE_NAME}; print({PIP_PACKAGE_NAME}.predict([[1,2,3,4]]))"',
            shell=True,
        )
    finally:
        print(
            subprocess.check_output(
                f"pip uninstall {PIP_PACKAGE_NAME} -y", shell=True
            )
        )


def test_whl_package(tmpdir, model_meta_saved_single):
    path = str(tmpdir)
    packager = WhlPackager(
        target=path, package_name=PIP_PACKAGE_NAME, version="1.0.0"
    )
    packager.package(model_meta_saved_single)
    files = os.listdir(tmpdir)
    assert len(files) == 1
    whl_path = files[0]
    assert whl_path.endswith(".whl")
    assert PIP_PACKAGE_NAME in whl_path
    assert "1.0.0" in whl_path
    subprocess.check_output(
        f"pip install {whl_path} --no-deps",
        shell=True,
        cwd=path,
    )
    try:
        subprocess.check_output(
            f'python -c "import {PIP_PACKAGE_NAME}; print({PIP_PACKAGE_NAME}.predict([[1,2,3,4]]))"',
            shell=True,
        )
    finally:
        subprocess.check_output(
            f"pip uninstall {PIP_PACKAGE_NAME} -y", shell=True
        )
