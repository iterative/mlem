from mlem.core.metadata import load_meta
from mlem.core.objects import MlemModel


def test_checkenv(runner, model_path_mlem_project):
    model_path, _ = model_path_mlem_project
    result = runner.invoke(
        ["checkenv", model_path],
    )
    assert result.exit_code == 0, (
        result.stdout,
        result.stderr,
        result.exception,
    )

    meta = load_meta(model_path, load_value=False, force_type=MlemModel)
    meta.requirements.__root__[0].version = "asdad"
    meta.update()

    result = runner.invoke(
        ["checkenv", model_path],
    )
    assert result.exit_code == 1, (
        result.stdout,
        result.stderr,
        result.exception,
    )
