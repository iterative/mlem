from typing import Optional

from mlem.core.objects import MlemModel
from mlem.runtime.server import Server
from mlem.ui import EMOJI_BUILD, echo, set_offset

from ..docker.base import DockerDaemon, DockerEnv, DockerRegistry
from ..docker.helpers import build_model_image


def build_k8s_docker(
    meta: MlemModel,
    image_name: str,
    registry: Optional[DockerRegistry],
    daemon: Optional[DockerDaemon],
    server: Server,
    platform: Optional[str] = "linux/amd64",
    # runners usually do not support arm64 images built on Mac M1 devices
):
    echo(EMOJI_BUILD + f"Creating docker image {image_name}")
    with set_offset(2):
        return build_model_image(
            meta,
            image_name,
            server,
            DockerEnv(registry=registry, daemon=daemon),
            tag=meta.meta_hash(),
            force_overwrite=True,
            platform=platform,
        )
