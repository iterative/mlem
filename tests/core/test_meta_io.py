import pytest

from mlem.core.meta_io import read
from tests.conftest import resource_path


@pytest.mark.parametrize(
    "url_path_pairs",
    [
        (resource_path(__file__, "file.txt"), "a"),
        (
            "github://iterative:example-mlem@main/README.md",
            "#",
        ),
    ],
)
def test_read(url_path_pairs):
    uri, start = url_path_pairs
    assert read(uri).startswith(start)


# @pytest.mark.parametrize(
#     "url_path_pairs",
#     [
#         (RESOURCES_PATH, {"file.txt": "a"}),
#         (
#             "https://github.com/mike0sv/ssci/tree/main/src/ssci/templates",
#             {"__init__.py": "", "docker-compose.yml": "version:"},
#         ),
#     ],
# )
# def test_blobs_from_path(url_path_pairs):
#     uri, files = url_path_pairs
#     blobs = blobs_from_path(uri)
#     assert set(blobs.blobs.keys()) == set(files.keys())
#     for key, blob in blobs.blobs.items():
#         assert (
#             blob.bytes().decode("utf8").startswith(files[key])
#         ), f"wrong contents for {key}"
