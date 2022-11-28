"""Scikit-Learn models support
Extension type: model

ModelType implementations for any sklearn-compatible classes as well as `Pipeline`
"""
from typing import Any, ClassVar, List, Optional, Union

import sklearn
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.pipeline import Pipeline

from mlem.core.hooks import IsInstanceHookMixin
from mlem.core.model import (
    ModelHook,
    ModelIO,
    ModelType,
    Signature,
    SimplePickleIO,
)
from mlem.core.requirements import InstallableRequirement, Requirements
from mlem.utils.module import get_object_base_module, get_object_requirements


class SklearnModel(ModelType, ModelHook, IsInstanceHookMixin):
    """ModelType implementation for `scikit-learn` models"""

    type: ClassVar[str] = "sklearn"
    valid_types: ClassVar = (RegressorMixin, ClassifierMixin)

    io: ModelIO = SimplePickleIO()
    """IO"""

    @classmethod
    def process(
        cls, obj: Any, sample_data: Optional[Any] = None, **kwargs
    ) -> ModelType:
        methods = {
            "predict": Signature.from_method(
                obj.predict, auto_infer=sample_data is not None, X=sample_data
            ),
        }
        if hasattr(obj, "predict_proba"):
            methods["predict_proba"] = Signature.from_method(
                obj.predict_proba,
                auto_infer=sample_data is not None,
                X=sample_data,
            )

        return SklearnModel(io=SimplePickleIO(), methods=methods).bind(obj)

    def get_requirements(self) -> Requirements:
        if not isinstance(self.io, SimplePickleIO):
            io_reqs: Union[Requirements, List] = get_object_requirements(
                self.io
            )
        else:
            io_reqs = []
        if get_object_base_module(self.model) is sklearn and not isinstance(
            self.model, Pipeline
        ):
            return (
                Requirements.resolve(
                    InstallableRequirement.from_module(sklearn)
                )
                + get_object_requirements(self.methods)
                + io_reqs
            )  # FIXME: https://github.com/iterative/mlem/issues/34 # optimize methods reqs

        # some sklearn compatible model (either from library or user code) - fallback
        return (
            super().get_requirements()
            + InstallableRequirement.from_module(sklearn)
            + io_reqs
        )


class SklearnPipelineType(SklearnModel):
    """ModelType implementation for `scikit-learn` pipelines"""

    valid_types: ClassVar = (Pipeline,)
    type: ClassVar = "sklearn_pipeline"

    @classmethod
    def process(
        cls, obj: Any, sample_data: Optional[Any] = None, **kwargs
    ) -> ModelType:
        mt = SklearnPipelineType(io=SimplePickleIO(), methods={}).bind(obj)
        predict = obj.predict
        predict_args = {"X": sample_data}
        if hasattr(predict, "__wrapped__"):
            predict = predict.__wrapped__
            predict_args["self"] = obj
        mt.methods["predict"] = Signature.from_method(
            predict, auto_infer=sample_data is not None, **predict_args
        )

        if hasattr(obj, "predict_proba"):
            predict_proba = obj.predict_proba
            predict_proba_args = {"X": sample_data}
            if hasattr(predict_proba, "__wrapped__"):
                predict_proba = predict_proba.__wrapped__
                predict_proba_args["self"] = obj
            mt.methods["predict_proba"] = Signature.from_method(
                predict_proba,
                auto_infer=sample_data is not None,
                **predict_proba_args
            )
        return mt
