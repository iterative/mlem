import os
import tempfile
from typing import Any, ClassVar, Optional

import catboost
from catboost import CatBoost, CatBoostClassifier, CatBoostRegressor
from fsspec import AbstractFileSystem

from mlem.core.artifacts import Artifacts
from mlem.core.model import ModelHook, ModelIO, ModelType, Signature
from mlem.core.requirements import InstallableRequirement, Requirements


class CatBoostModelIO(ModelIO):
    """
    :class:`mlem.core.model.ModelIO` for CatBoost models.
    """

    type: ClassVar[str] = "catboost_io"
    classifier_file_name = "clf.cb"
    regressor_file_name = "rgr.cb"

    def dump(self, fs: AbstractFileSystem, path, model) -> Artifacts:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_name = self._get_model_file_name(model)
            model_path = os.path.join(tmpdir, model_name)
            model.save_model(model_path)
            fs.put(model_path, os.path.join(path, model_name))
            return [os.path.join(path, model_name)]

    def load(self, fs: AbstractFileSystem, path):
        """
        Loads `catboost.CatBoostClassifier` or `catboost.CatBoostRegressor` instance from path

        """
        if fs.exists(os.path.join(path, self.classifier_file_name)):
            model_type = CatBoostClassifier
        else:
            model_type = CatBoostRegressor

        model = model_type()
        with fs.open(
            os.path.join(path, self._get_model_file_name(model))
        ) as f:
            model.load_model(stream=f)
        return model

    def _get_model_file_name(self, model):
        if isinstance(model, CatBoostClassifier):
            return self.classifier_file_name
        return self.regressor_file_name


class CatBoostModel(ModelType, ModelHook):
    """
    :class:`mlem.core.model.ModelType` for CatBoost models.
    `.model` attribute is a `catboost.CatBoostClassifier` or `catboost.CatBoostRegressor` instance
    """

    type: ClassVar[str] = "catboost"
    io: ModelIO = CatBoostModelIO()
    model: ClassVar[Optional[CatBoost]]

    @classmethod
    def is_object_valid(cls, obj: Any) -> bool:
        return isinstance(obj, (CatBoostClassifier, CatBoostRegressor))

    @classmethod
    def process(
        cls, obj: Any, sample_data: Optional[Any] = None, **kwargs
    ) -> ModelType:
        model = CatBoostModel(model=obj, methods={})
        methods = {
            "predict": Signature.from_method(
                model.predict,
                auto_infer=sample_data is not None,
                data=sample_data,
            ),
            "catboost_predict": Signature.from_method(
                obj.predict,
                auto_infer=sample_data is not None,
                data=sample_data,
            ),
        }
        if isinstance(obj, CatBoostClassifier):
            methods["predict_proba"] = Signature.from_method(
                model.predict_proba,
                auto_infer=sample_data is not None,
                data=sample_data,
            )
            methods["catboost_predict_proba"] = Signature.from_method(
                obj.predict_proba,
                auto_infer=sample_data is not None,
                X=sample_data,
            )
        model.methods = methods
        return model

    def predict(self, data):
        return self.model.predict(data)

    def predict_proba(self, data):
        if not isinstance(self.model, CatBoostClassifier):
            raise ValueError(
                "Not valid type of model for predict_proba method"
            )
        return self.model.predict_proba(data)

    def get_requirements(self) -> Requirements:
        return super().get_requirements() + InstallableRequirement.from_module(
            catboost
        )
