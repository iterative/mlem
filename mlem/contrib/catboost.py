"""Catboost Models Support
Extension type: model

Implementations of ModelType and ModelIO for `CatBoostClassifier` and `CatBoostRegressor`
"""
import os
import posixpath
import tempfile
from enum import Enum
from typing import Any, ClassVar, Optional

import catboost
from catboost import CatBoost, CatBoostClassifier, CatBoostRegressor

from mlem.core.artifacts import Artifacts, Storage
from mlem.core.hooks import IsInstanceHookMixin
from mlem.core.model import ModelHook, ModelIO, ModelType, Signature
from mlem.core.requirements import InstallableRequirement, Requirements


class CBType(str, Enum):
    classifier = "clf"
    regressor = "reg"


class CatBoostModelIO(ModelIO):
    """
    :class:`mlem.core.model.ModelIO` for CatBoost models.
    """

    type: ClassVar[str] = "catboost_io"
    classifier_file_name: ClassVar = "clf.cb"
    """Filename for catboost classifier"""
    regressor_file_name: ClassVar = "rgr.cb"
    """Filename for catboost classifier"""
    model_type: CBType = CBType.regressor
    """Type of catboost model"""

    def dump(self, storage: Storage, path, model) -> Artifacts:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_name = self._get_model_file_name(model)
            model_path = os.path.join(tmpdir, model_name)
            model.save_model(model_path)
            return {
                self.art_name: storage.upload(
                    model_path, posixpath.join(path, model_name)
                )
            }

    def load(self, artifacts: Artifacts):
        """
        Loads `catboost.CatBoostClassifier` or `catboost.CatBoostRegressor` instance from path

        """
        if len(artifacts) != 1:
            raise ValueError(
                f"Invalid artifacts: should be one {self.classifier_file_name} or {self.regressor_file_name} file"
            )
        if self.model_type == CBType.classifier:
            model_type = CatBoostClassifier
        else:
            model_type = CatBoostRegressor

        model = model_type()
        with artifacts[self.art_name].open() as f:
            model.load_model(stream=f)
        return model

    def _get_model_file_name(self, model):
        if isinstance(model, CatBoostClassifier):
            return self.classifier_file_name
        return self.regressor_file_name

    class Config:
        use_enum_values = True


class CatBoostModel(ModelType, ModelHook, IsInstanceHookMixin):
    """
    :class:`mlem.core.model.ModelType` for CatBoost models.
    `.model` attribute is a `catboost.CatBoostClassifier` or `catboost.CatBoostRegressor` instance
    """

    type: ClassVar[str] = "catboost"
    model: ClassVar[Optional[CatBoost]]
    valid_types: ClassVar = (CatBoostClassifier, CatBoostRegressor)

    io: ModelIO = CatBoostModelIO()

    @classmethod
    def process(
        cls, obj: Any, sample_data: Optional[Any] = None, **kwargs
    ) -> ModelType:
        model = CatBoostModel(model=obj, methods={})
        methods = {
            "predict": Signature.from_method(
                obj.predict,
                auto_infer=sample_data is not None,
                data=sample_data,
            ),
        }
        if isinstance(obj, CatBoostClassifier):
            model.io.model_type = CBType.classifier
            methods["predict_proba"] = Signature.from_method(
                obj.predict_proba,
                auto_infer=sample_data is not None,
                X=sample_data,
            )
        model.methods = methods
        return model

    def get_requirements(self) -> Requirements:
        return super().get_requirements() + InstallableRequirement.from_module(
            catboost
        )
