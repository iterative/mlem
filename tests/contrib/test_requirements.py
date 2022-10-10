from mlem.contrib.requirements import RequirementsBuilder


def test_build_reqs(tmp_path, model_meta):
    path = str(tmp_path / "reqs.txt")
    builder = RequirementsBuilder(target=path)
    builder.build(model_meta)
    with open(path, "r", encoding="utf-8") as f:
        assert model_meta.requirements.to_pip() == f.read().splitlines()


def test_build_requirements_should_print_with_no_path(capsys, model_meta):
    builder = RequirementsBuilder()
    builder.build(model_meta)
    captured = capsys.readouterr()
    assert captured.out == "\n".join(model_meta.requirements.to_pip()) + "\n"
