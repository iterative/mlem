import click

from mlem.cli.main import mlem_command, with_meta
from mlem.core.objects import ModelMeta


@mlem_command("serve")
@with_meta("model", force_cls=ModelMeta)
@click.argument("server", default="fastapi")
def serve(model: ModelMeta, server: str):
    """Serve selected model."""
    from mlem.api.commands import serve

    click.echo("Serving")
    serve(model, server)
