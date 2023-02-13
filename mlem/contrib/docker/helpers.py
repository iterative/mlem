from mlem.core.objects import MlemModel
from mlem.runtime.server import Server

from . import DockerImageBuilder
from .base import (
    DockerBuildArgs,
    DockerDaemon,
    DockerImage,
    DockerImageOptions,
    DockerRegistry,
)


def build_model_image(
    model: MlemModel,
    name: str,
    server: Server = None,
    daemon: DockerDaemon = None,
    registry: DockerRegistry = None,
    tag: str = "latest",
    repository: str = None,
    force_overwrite: bool = True,
    push: bool = True,
    **build_args
) -> DockerImage:
    registry = registry or DockerRegistry()
    daemon = daemon or DockerDaemon()
    image = DockerImageOptions(
        name=name, tag=tag, repository=repository, registry=registry
    )
    builder = DockerImageBuilder(
        server=server,
        args=DockerBuildArgs(**build_args),
        image=image,
        daemon=daemon,
        force_overwrite=force_overwrite,
        push=push,
    )
    return builder.build(model)
