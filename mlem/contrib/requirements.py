"""Requirements support
Extension type: build

MlemBuilder implementation for `Requirements` which includes
installable, conda, unix, custom, file etc. based requirements.
"""
import logging
from typing import ClassVar, Optional

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
    """Target path for requirements"""
    req_type: str = "installable"
    """Type of requirements, example: unix"""

    @validator("req_type")
    def get_req_type(cls, req_type):  # pylint: disable=no-self-argument
        if req_type not in list_implementations(Requirement):
            raise ValueError(
                f"req_type {req_type} is not valid. Allowed options are: {list_implementations(Requirement)}"
            )
        return req_type

    def build(self, obj: MlemModel):
        req_type_cls = load_impl_ext(Requirement.abs_name, self.req_type)
        assert issubclass(req_type_cls, Requirement)
        reqs = obj.requirements.of_type(req_type_cls)
        if self.target is None:
            reqs_representation = [r.get_repr() for r in reqs]
            requirement_string = " ".join(reqs_representation)
            print(requirement_string)
        else:
            echo(EMOJI_PACK + "Materializing requirements...")
            req_type_cls.materialize(reqs, self.target)
            echo(EMOJI_OK + f"Materialized to {self.target}!")
