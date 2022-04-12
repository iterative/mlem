from typer import Argument, Typer

from mlem.cli.main import MlemGroupSection, app, mlem_command
from mlem.ext import MLEM_ENTRY_POINT, find_implementations, load_entrypoints
from mlem.ui import echo

dev = Typer(name="dev", cls=MlemGroupSection("common"), hidden=True)
app.add_typer(dev)


@dev.callback()
def dev_callback():
    """Developer utility tools"""


@mlem_command(parent=dev, aliases=["fi"])
def find_implementations_diff(
    root: str = Argument(MLEM_ENTRY_POINT, help="root entry point")
):
    """Loads `root` module or package and finds implementations of MLEM base classes
    Shows differences between what was found and what is registered in entrypoints

    Examples:
        $ mlem dev fi
    """
    exts = {e.entry for e in load_entrypoints().values()}
    impls = set(find_implementations(root)[MLEM_ENTRY_POINT])
    extra = exts.difference(impls)
    if extra:
        echo("Remove implementations:")
        echo("\n".join(extra))
    new = impls.difference(exts)
    if new:
        echo("Add implementations:")
        echo("\n".join(new))
