from mlem.core.objects import ModelMeta
from mlem.pack.docker import DockerImagePackager
from mlem.pack.docker.base import DockerEnv, DockerImage, DockerBuildArgs
from mlem.runtime.server.base import Server


def build_model_image(model: ModelMeta, name: str, server: Server = None, env: DockerEnv = None, tag: str = 'latest',
                      repository: str = None, force_overwrite: bool = False, **build_args):
    packager = DockerImagePackager(
        server=server, args=DockerBuildArgs(**build_args), image=DockerImage(name=name, tag=tag, repository=repository),
        env=env, force_overwrite=force_overwrite
    )
    return packager.package(model, name)
