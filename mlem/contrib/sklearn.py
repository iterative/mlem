from typing import Any, ClassVar, Optional

import sklearn
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.pipeline import Pipeline

from mlem.constants import (
    PREDICT_ARG_NAME,
    PREDICT_METHOD_NAME,
    PREDICT_PROBA_METHOD_NAME,
)
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
    """
    :class:`mlem.core.model.ModelType implementation for `scikit-learn` models
    """

    type: ClassVar[str] = "sklearn"
    io: ModelIO = SimplePickleIO()
    valid_types: ClassVar = (RegressorMixin, ClassifierMixin)

    @classmethod
    def process(
        cls, obj: Any, sample_data: Optional[Any] = None, **kwargs
    ) -> ModelType:
        sklearn_predict = Signature.from_method(
            obj.predict, sample_data is not None, X=sample_data
        )
        predict = sklearn_predict.copy()
        predict.args = [predict.args[0].copy()]
        predict.args[0].name = PREDICT_ARG_NAME
        methods = {
            "sklearn_predict": sklearn_predict,
            PREDICT_METHOD_NAME: predict,
        }
        if hasattr(obj, "predict_proba"):
            sklearn_predict_proba = Signature.from_method(
                obj.predict_proba, sample_data is not None, X=sample_data
            )
            predict_proba = sklearn_predict_proba.copy()
            predict_proba.args = [predict_proba.args[0].copy()]
            predict_proba.args[0].name = PREDICT_ARG_NAME
            methods["sklearn_predict_proba"] = sklearn_predict_proba
            methods[PREDICT_PROBA_METHOD_NAME] = predict_proba

        return SklearnModel(io=SimplePickleIO(), methods=methods).bind(obj)

    def get_requirements(self) -> Requirements:
        if get_object_base_module(self.model) is sklearn:
            return Requirements.resolve(
                InstallableRequirement.from_module(sklearn)
            ) + get_object_requirements(
                self.methods
            )  # FIXME: https://github.com/iterative/mlem/issues/34 # optimize methods reqs

        # some sklearn compatible model (either from library or user code) - fallback
        return super().get_requirements() + InstallableRequirement.from_module(
            sklearn
        )


class SklearnPipelineType(SklearnModel):
    valid_types: ClassVar = (Pipeline,)
    type: ClassVar = "sklearn_pipeline"

    @classmethod
    def process(
        cls, obj: Any, sample_data: Optional[Any] = None, **kwargs
    ) -> ModelType:
        mt = SklearnModel(io=SimplePickleIO(), methods={}).bind(obj)
        predict = obj.predict
        predict_args = {"X": sample_data}
        if hasattr(predict, "__wrapped__"):
            predict = predict.__wrapped__
            predict_args["self"] = obj
        sk_predict_sig = Signature.from_method(
            predict, auto_infer=True, **predict_args
        )
        mt.methods["sklearn_predict"] = sk_predict_sig
        predict_sig = sk_predict_sig.copy()
        predict_sig.args[0].name = "data"
        predict_sig.varkw = None
        mt.methods[PREDICT_METHOD_NAME] = predict_sig
        return mt
