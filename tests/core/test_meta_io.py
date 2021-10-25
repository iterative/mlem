import pytest

from mlem.core.meta_io import read
from tests.conftest import resource_path


@pytest.mark.parametrize(
    "url_path_pairs",
    [
        (resource_path(__file__, "file.txt"), "a"),
        (
            "github://iterative:mlem-test@main/README.md",
            "#",
        ),
        ("https://github.com/iterative/mlem-test/README.md", "#"),
    ],
)
def test_read(url_path_pairs):
    uri, start = url_path_pairs
    assert read(uri).startswith(start)
