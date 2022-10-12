import logging
import os
from typing import ClassVar, Optional

from mlem.core.objects import MlemBuilder, MlemModel
from mlem.core.requirements import Requirements
from mlem.ui import EMOJI_OK, EMOJI_PACK, echo

REQUIREMENTS = "requirements.txt"

logger = logging.getLogger(__name__)


class RequirementsBuilder(MlemBuilder):
    """MlemBuilder implementation for building requirements"""

    type: ClassVar = "requirements"

    target: Optional[str] = None
    """Target path for the requirements.txt file"""
    platform: str = "pip"
    """Type of requirements, example: unix"""

    def write_requirements_file(self, requirements: Requirements):
        requirement_string = "\n".join(requirements.to_pip())
        if self.target is None:
            print(requirement_string)
        else:
            echo(EMOJI_PACK + "Generating requirements file...")
            with open(os.path.join(self.target), "w", encoding="utf8") as req:
                logger.debug(
                    "Auto-determined requirements for model: %s.",
                    requirements.to_pip(),
                )
                req.write(requirement_string + "\n")
                echo(EMOJI_OK + f"{self.target} generated!")

    def build(self, obj: MlemModel):
        if self.platform == "unix":
            unix_reqs = obj.requirements.to_unix()
            requirement_string = "\n".join(unix_reqs)
            if requirement_string:
                print(requirement_string)
        elif self.platform == "pip":
            self.write_requirements_file(obj.requirements)
