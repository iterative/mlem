import pytest

from mlem.utils.path import make_posix


@pytest.mark.parametrize(
    "path, result",
    [
        ("relative/posix/path", "relative/posix/path"),
        ("relative\\nt\\path", "relative/nt/path"),
        ("relative/nt/path", "relative/nt/path"),
        ("/abs/posix/path", "/abs/posix/path"),
        ("c:\\abs\\nt\\path", "c:/abs/nt/path"),
        ("c:/abs/nt/path", "c:/abs/nt/path"),
        ("/aaa\\bbb", "/aaa/bbb"),
        ("mixed\\nt/path", "mixed/nt/path"),
        ("\\aaa\\bbb", "/aaa/bbb"),
        ("\\aaa/bbb", "/aaa/bbb"),
        ("c:\\mixed/nt/path", "c:/mixed/nt/path"),
        ("c:/mixed\\nt\\path", "c:/mixed/nt/path"),
        ("", ""),
        ("aaa", "aaa"),
    ],
)
def test_make_posix(path, result):
    assert make_posix(path) == result
