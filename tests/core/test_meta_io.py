import pytest

from mlem.core.meta_io import read
from tests.conftest import (
    MLEM_TEST_REPO_NAME,
    MLEM_TEST_REPO_ORG,
    resource_path,
)


@pytest.mark.parametrize(
    "url_path_pairs",
    [
        (resource_path(__file__, "file.txt"), "a"),
        (
            f"github://{MLEM_TEST_REPO_ORG}:{MLEM_TEST_REPO_NAME}@main/README.md",
            "#",
        ),
        (
            f"https://github.com/{MLEM_TEST_REPO_ORG}/{MLEM_TEST_REPO_NAME}/README.md",
            "#",
        ),
    ],
)
def test_read(url_path_pairs):
    uri, start = url_path_pairs
    assert read(uri).startswith(start)
