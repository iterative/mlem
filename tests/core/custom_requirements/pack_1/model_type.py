import typing
from typing import Any, Optional

import numpy as np
from pack_1 import model

from mlem.core.model import ModelHook, ModelType, SimplePickleIO


class TestModelType(ModelType, ModelHook):
    @classmethod
    def process(
        cls, obj: Any, sample_data: Optional[Any] = None, **kwargs
    ) -> ModelType:
        return TestModelType(io=SimplePickleIO(), methods={})

    @classmethod
    def is_object_valid(cls, obj: Any) -> bool:
        return isinstance(obj, model.TestM)

    def m1(self) -> typing.Dict[str, str]:
        return {}

    def m2(self, data: np.array):  # pylint: disable=unused-argument
        return
