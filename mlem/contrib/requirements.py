import logging
import os
from typing import ClassVar, List, Optional

from pydantic import validator

from mlem.core.base import load_impl_ext
from mlem.core.objects import MlemBuilder, MlemModel
from mlem.core.requirements import Requirement
from mlem.ui import EMOJI_OK, EMOJI_PACK, echo
from mlem.utils.entrypoints import list_implementations

REQUIREMENTS = "requirements.txt"

logger = logging.getLogger(__name__)


class RequirementsBuilder(MlemBuilder):
    """MlemBuilder implementation for building requirements"""

    type: ClassVar = "requirements"

    target: Optional[str] = None
    """Target path for the requirements.txt file"""
    req_type: str = "installable"
    """Type of requirements, example: unix"""

    @validator("req_type")
    def get_req_type(cls, req_type):  # pylint: disable=no-self-argument
        if req_type not in list_implementations(Requirement):
            raise ValueError(
                f"req_type {req_type} is not valid. Allowed options are: {list_implementations(Requirement)}"
            )
        return req_type

    def write_requirements_file(self, reqs: List[str]):
        requirement_string = "\n".join(reqs)
        if self.target is None:
            print(requirement_string)
        else:
            echo(EMOJI_PACK + "Generating requirements file...")
            with open(os.path.join(self.target), "w", encoding="utf8") as fp:
                fp.write(requirement_string + "\n")
                echo(EMOJI_OK + f"{self.target} generated!")

    def build(self, obj: MlemModel):
        req_type_cls = load_impl_ext(Requirement.abs_name, self.req_type)
        assert issubclass(req_type_cls, Requirement)
        reqs = obj.requirements.of_type(req_type_cls)
        reqs_representation = [r.get_repr() for r in reqs]
        self.write_requirements_file(reqs_representation)
