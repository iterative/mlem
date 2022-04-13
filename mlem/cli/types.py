from typing import Optional

from typer import Argument

from mlem.cli.main import mlem_command
from mlem.core.base import MlemObject
from mlem.core.objects import MlemMeta
from mlem.ext import list_implementations
from mlem.ui import EMOJI_BASE, bold, echo


@mlem_command("types", hidden=True)
def list_types(
    subtype: Optional[str] = Argument(
        None,
        help="Subtype to list implementations. List subtypes if not provided",
    ),
    meta_type: Optional[str] = Argument(None, help="Type of `meta` subtype"),
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
            echo(EMOJI_BASE + bold(at.abs_name) + ":")
            echo(
                f"\tBase class: {at.__module__}.{at.__name__}\n\t{at.__doc__.strip()}"
            )
    elif subtype == MlemMeta.abs_name:
        if meta_type is None:
            echo(list(MlemMeta.non_abstract_subtypes().keys()))
        else:
            echo(
                list_implementations(
                    MlemMeta, MlemMeta.non_abstract_subtypes()[meta_type]
                )
            )
    else:
        echo(list_implementations(subtype))
