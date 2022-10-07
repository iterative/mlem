from mlem.contrib.venv import VenvBuilder


def test_build_venv(tmp_path, model_meta):
    path = str(tmp_path / "venv")
    builder = VenvBuilder(target=path)
    context = builder.build(model_meta)
    installed_pkgs = (
        builder.get_installed_packages(context).decode().splitlines()
    )
    for each_req in model_meta.requirements.to_pip():
        assert each_req in installed_pkgs
