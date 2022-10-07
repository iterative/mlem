from mlem.contrib.requirements import RequirementsBuilder


def test_build_reqs(tmp_path, model_meta):
    path = str(tmp_path / "reqs.txt")
    builder = RequirementsBuilder(target=path)
    builder.build(model_meta)
    with open(path, "r") as f:
        assert model_meta.requirements.to_pip() == f.read().splitlines()
