from typing import Optional

from typer import Argument

from mlem.cli.main import mlem_command, option_repo, option_rev
from mlem.core.metadata import load_meta
from mlem.core.objects import MlemDataset, MlemModel
from mlem.ui import EMOJI_OK, echo


@mlem_command("checkenv", hidden=True)
def checkenv(
    path: str = Argument(..., help="Path to object"),
    repo: Optional[str] = option_repo,
    rev: Optional[str] = option_rev,
):
    """Check that current environment satisfies object requrements

    Examples:
        Check local object
        $ mlem checkenv mymodel

        Check remote object
        $ mlem checkenv https://github.com/iterative/example-mlem/models/logreg
    """
    meta = load_meta(path, repo, rev, follow_links=True, load_value=False)
    if isinstance(meta, (MlemModel, MlemDataset)):
        meta.checkenv()
    echo(EMOJI_OK + "Requirements are satisfied!")
