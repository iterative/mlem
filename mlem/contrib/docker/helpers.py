from mlem.core.objects import MlemModel
from mlem.runtime.server import Server

from . import DockerImageBuilder
from .base import DockerBuildArgs, DockerEnv, DockerImage


def build_model_image(
    model: MlemModel,
    name: str,
    server: Server = None,
    env: DockerEnv = None,
    tag: str = "latest",
    repository: str = None,
    force_overwrite: bool = False,
    push: bool = True,
    **build_args
) -> DockerImage:
    env = env or DockerEnv()
    image = DockerImage(
        name=name, tag=tag, repository=repository, registry=env.registry
    )
    builder = DockerImageBuilder(
        server=server,
        args=DockerBuildArgs(**build_args),
        image=image,
        env=env,
        force_overwrite=force_overwrite,
        push=push,
    )
    builder.build(model)
    return builder.image
