from typing import Optional

from typer import Argument

from mlem.cli.main import mlem_command
from mlem.core.base import MlemObject
from mlem.ext import list_implementations
from mlem.ui import EMOJI_CASE, echo


@mlem_command("types", hidden=True)
def list_types(
    subtype: Optional[str] = Argument(
        None,
        help="Subtype to list implementations. List subtypes if not provided",
    )
):
    """List MLEM types implementations available in current env.
    If subtype is not provided, list ABCs

    Examples:
        List ABCs
        $ mlem types

        List available server implementations
        $ mlem types server
    """
    if subtype is None:
        for at in MlemObject.abs_types:
            echo(EMOJI_CASE + at.abs_name + ":")
            echo(
                f"\tBase class: {at.__module__}.{at.__name__}\n\t{at.__doc__.strip()}"
            )

    else:
        echo(list_implementations(subtype))
