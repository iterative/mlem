from mlem.config import MlemConfig
from tests.conftest import resource_path


def test_loading_yaml(mocker):
    mocker.patch(
        "mlem.utils.root.find_mlem_root", return_value=resource_path(__file__)
    )
    config = MlemConfig()
    assert config.ADDITIONAL_EXTENSIONS == ["ext1"]
