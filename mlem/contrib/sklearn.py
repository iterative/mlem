from typing import Any, ClassVar, Dict

import sklearn
from sklearn.base import ClassifierMixin, RegressorMixin

from mlem.core.dataset_type import DatasetAnalyzer, UnspecifiedDatasetType
from mlem.core.model import (
    Argument,
    ModelHook,
    ModelIO,
    ModelType,
    Signature,
    SimplePickleIO,
)
from mlem.core.requirements import InstallableRequirement, Requirements
from mlem.utils.module import get_object_base_module, get_object_requirements


class SklearnModel(ModelType, ModelHook):
    """
    :class:`mlem.core.model.ModelType implementation for `scikit-learn` models
    """

    type: ClassVar[str] = "sklearn"
    io: ModelIO = SimplePickleIO()

    @classmethod
    def is_object_valid(cls, obj: Any) -> bool:
        return isinstance(obj, (RegressorMixin, ClassifierMixin))

    @classmethod
    def process(cls, obj: Any, **kwargs) -> "SklearnModel":
        test_data = kwargs.get("test_data")
        method_names = ["predict"]
        if isinstance(obj, ClassifierMixin):
            method_names.append("predict_proba")
        if test_data is None:

            methods: Dict[str, Signature] = {
                m: Signature(
                    name=m,
                    args=[Argument(key="X", type=UnspecifiedDatasetType())],
                    returns=UnspecifiedDatasetType(),
                )
                for m in method_names
            }

        else:
            methods = {
                m: Signature(
                    name=m,
                    args=[
                        Argument(
                            key="X", type=DatasetAnalyzer.analyze(test_data)
                        )
                    ],
                    returns=DatasetAnalyzer.analyze(
                        getattr(obj, m)(test_data)
                    ),
                )
                for m in method_names
            }

        return SklearnModel(io=SimplePickleIO(), methods=methods).bind(obj)

    def get_requirements(self) -> Requirements:
        if get_object_base_module(self.model) is sklearn:
            return Requirements.resolve(
                InstallableRequirement.from_module(sklearn)
            ) + get_object_requirements(
                self.methods
            )  # FIXME: https://github.com/iterative/mlem/issues/34 # optimize methods reqs

        # some sklearn compatible model (either from library or user code) - fallback
        return super().get_requirements()
