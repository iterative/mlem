from pathlib import Path
from unittest.mock import ANY, patch

from mlem.contrib.flyio.meta import FlyioApp
from mlem.contrib.flyio.utils import FlyioStatusModel


def test_flyio_create_app(tmp_path: Path):
    flyio_app = FlyioApp(org="org", app_name="test")
    flyio_app.dump(str(tmp_path))
    state = flyio_app.get_state()
    status = FlyioStatusModel(
        Name="test", Hostname="fly.io", Status="Deployed"
    )

    with patch("mlem.contrib.flyio.meta.run_flyctl") as run_flyctl:
        with patch("mlem.contrib.flyio.meta.read_fly_toml") as read_fly_toml:
            with patch("mlem.contrib.flyio.meta.get_status") as get_status:
                with patch("mlem.contrib.flyio.meta.FlyioApp._build_in_dir"):
                    get_status.return_value = status
                    read_fly_toml.return_value = ""
                    flyio_app.deploy(state)

    run_flyctl.assert_called_once_with(
        "launch",
        workdir=ANY,
        kwargs={
            "auto-confirm": True,
            "reuse-app": True,
            "region": "lax",
            "no-deploy": True,
            "name": "test",
            "org": "org",
        },
    )
