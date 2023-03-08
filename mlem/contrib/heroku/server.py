import logging
import os
from typing import ClassVar

from pydantic import validator

from mlem.runtime import Interface
from mlem.runtime.server import Server

logger = logging.getLogger(__name__)


class HerokuServer(Server):
    """Special FastAPI server to pickup port from env PORT"""

    type: ClassVar = "_heroku"
    server: Server

    @validator("server")
    @classmethod
    def server_validator(cls, value: Server):
        if value.port_field is None:
            raise ValueError(
                f"{value} does not have port field and can not be exposed on heroku"
            )
        return value

    def serve(self, interface: Interface):
        assert self.server.port_field is not None  # ensured by validator
        setattr(self.server, self.server.port_field, int(os.environ["PORT"]))
        logger.info(
            "Switching port to %s",
            getattr(self.server, self.server.port_field),
        )
        return self.server.serve(interface)
