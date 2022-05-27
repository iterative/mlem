import os.path
import subprocess

import numpy as np
import pytest

from mlem.core.metadata import load_meta, save
from mlem.core.objects import MlemModel


@pytest.fixture
def script_code():
    with open(
        os.path.join(os.path.dirname(__file__), "shell_reqs.py"),
        encoding="utf8",
    ) as f:
        return f.read()


exec_param = pytest.mark.parametrize("executable", ["python", "ipython"])


@exec_param
@pytest.mark.xfail
def test_cmd(tmpdir, script_code, executable):
    res = subprocess.check_call([executable, "-c", script_code], cwd=tmpdir)
    assert res == 0
    save("a", os.path.join(tmpdir, "data"))
    subprocess.check_call(["mlem", "apply", "model", "data"], cwd=tmpdir)

    meta = load_meta(os.path.join(tmpdir, "model"), force_type=MlemModel)
    assert len(meta.requirements.__root__) == 1
    assert meta.requirements.to_pip() == [f"numpy=={np.__version__}"]


@exec_param
@pytest.mark.xfail
def test_pipe(tmpdir, script_code, executable):
    res = subprocess.check_call(
        f"echo '{script_code}' | {executable}", cwd=tmpdir, shell=True
    )
    assert res == 0
    save("a", os.path.join(tmpdir, "data"))
    subprocess.check_call(["mlem", "apply", "model", "data"], cwd=tmpdir)

    meta = load_meta(os.path.join(tmpdir, "model"), force_type=MlemModel)
    assert len(meta.requirements.__root__) == 1
    assert meta.requirements.to_pip() == [f"numpy=={np.__version__}"]


@exec_param
@pytest.mark.xfail
def test_pipe_iter(tmpdir, script_code, executable):
    with subprocess.Popen(executable, stdin=subprocess.PIPE) as proc:
        for line in script_code.splitlines(keepends=True):
            proc.stdin.write(line.encode("utf8"))
        proc.communicate(b"exit()")
        assert proc.returncode == 0
    save("a", os.path.join(tmpdir, "data"))
    subprocess.check_call(["mlem", "apply", "model", "data"], cwd=tmpdir)

    meta = load_meta(os.path.join(tmpdir, "model"), force_type=MlemModel)
    assert len(meta.requirements.__root__) == 1
    assert meta.requirements.to_pip() == [f"numpy=={np.__version__}"]


@exec_param
@pytest.mark.xfail
def test_script(tmpdir, script_code, executable):
    with open(tmpdir / "script.py", "w", encoding="utf8") as f:
        f.write(script_code)
    res = subprocess.check_call([executable, "script.py"], cwd=tmpdir)
    assert res == 0
    save("a", os.path.join(tmpdir, "data"))
    subprocess.check_call(["mlem", "apply", "model", "data"], cwd=tmpdir)

    meta = load_meta(os.path.join(tmpdir, "model"), force_type=MlemModel)
    assert len(meta.requirements.__root__) == 1
    assert meta.requirements.to_pip() == [f"numpy=={np.__version__}"]
