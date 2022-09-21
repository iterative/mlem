import dvc
import importlib_metadata
from packaging import version

from setup import extras


def test_dvc_extras():
    # previous to 2.15 DVC had a typo in extras
    if version.parse(dvc.__version__) > version.parse("2.15"):
        # importlib_metadata checks the locally installed package,
        # so this may pass locally, but fail in CI
        correct_extras = {
            f"dvc-{e}": [f"dvc[{e}]~=2.0"]
            for e in importlib_metadata.metadata("dvc").get_all(
                "Provides-Extra"
            )
            if e not in {"all", "dev", "terraform", "tests", "testing"}
        }
        specified_extras = {
            e: l for e, l in extras.items() if e[: len("dvc-")] == "dvc-"
        }
        assert correct_extras == specified_extras
