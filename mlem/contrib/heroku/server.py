import logging
import os
from typing import ClassVar

from mlem.contrib.fastapi import FastAPIServer
from mlem.runtime import Interface

logger = logging.getLogger(__name__)


class HerokuServer(FastAPIServer):
    """Special FastAPI server to pickup port from env PORT"""

    type: ClassVar = "_heroku"

    def serve(self, interface: Interface):
        self.port = int(os.environ["PORT"])
        logger.info("Switching port to %s", self.port)
        return super().serve(interface)
