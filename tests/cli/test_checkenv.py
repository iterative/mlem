from mlem.core.metadata import load_meta
from mlem.core.objects import ModelMeta


def test_checkenv(runner, model_path_mlem_repo):
    model_path, _ = model_path_mlem_repo
    result = runner.invoke(
        ["checkenv", model_path],
    )
    assert result.exit_code == 0, (result.output, result.exception)

    meta = load_meta(model_path, load_value=False, force_type=ModelMeta)
    meta.requirements.__root__[0].version = "asdad"
    meta.update()

    result = runner.invoke(
        ["checkenv", model_path],
    )
    assert result.exit_code == 1, (result.output, result.exception)
