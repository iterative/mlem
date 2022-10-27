from typer import Argument, Typer

from mlem.cli.main import app, mlem_command, mlem_group
from mlem.ui import echo
from mlem.utils.entrypoints import (
    MLEM_ENTRY_POINT,
    find_abc_implementations,
    load_entrypoints,
)

dev = Typer(name="dev", cls=mlem_group("common"), hidden=True)
app.add_typer(dev)


@dev.callback()
def dev_callback():
    """Developer utility tools

    Documentation: <https://mlem.ai/doc/contributing/core>
    """


@mlem_command(parent=dev, aliases=["fi"])
def find_implementations_diff(
    root: str = Argument(MLEM_ENTRY_POINT, help="root entry point")
):
    """Loads `root` module or package and finds implementations of MLEM base classes
    Shows differences between what was found and what is registered in entrypoints

    Documentation: <https://mlem.ai/doc/contributing/core>
    """
    exts = {e.entry for e in load_entrypoints().values()}
    impls = set(find_abc_implementations(root)[MLEM_ENTRY_POINT])
    extra = exts.difference(impls)
    if extra:
        echo("Remove implementations:")
        echo("\n".join(extra))
    new = impls.difference(exts)
    if new:
        echo("Add implementations:")
        echo("\n".join(new))
