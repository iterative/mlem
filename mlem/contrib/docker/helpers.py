from mlem.core.objects import ModelMeta
from mlem.runtime.server.base import Server

from . import DockerImagePackager
from .base import DockerBuildArgs, DockerEnv, DockerImage


def build_model_image(
    model: ModelMeta,
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
    packager = DockerImagePackager(
        server=server,
        args=DockerBuildArgs(**build_args),
        image=image,
        env=env,
        force_overwrite=force_overwrite,
        push=push,
    )
    packager.package(model, image.uri)
    return packager.image
