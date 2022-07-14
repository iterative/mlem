import importlib_metadata

from setup import extras


def test_dvc_extras():

    # importlib_metadata checks the locally installed package,
    # so this may pass locally, but fail in CI
    for e in importlib_metadata.metadata("dvc").get_all("Provides-Extra"):
        if e not in {"all", "dev", "terraform", "tests"}:
            assert extras[f"dvc-{e}"] == [f"dvc[{e}]~=2.0"]
