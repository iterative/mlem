import os
import tempfile
from typing import Any, ClassVar

import catboost
from catboost import CatBoostClassifier, CatBoostRegressor
from fsspec import AbstractFileSystem

from mlem.core.artifacts import Artifacts
from mlem.core.dataset_type import UnspecifiedDatasetType
from mlem.core.model import Argument, ModelHook, ModelIO, ModelType, Signature
from mlem.core.requirements import LibRequirementsMixin


class CatBoostModelIO(ModelIO):
    """
    :class:`mlem.core.model.ModelIO` for CatBoost models.
    """

    type: ClassVar = "catboost_io"
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


class CatBoostModel(ModelType, ModelHook, LibRequirementsMixin):
    """
    :class:`mlem.core.model.ModelType` for CatBoost models.
    `.model` attribute is a `catboost.CatBoostClassifier` or `catboost.CatBoostRegressor` instance
    """

    libraries: ClassVar = [catboost]
    type: ClassVar = "catboost"
    io: ModelIO = CatBoostModelIO()

    @classmethod
    def is_object_valid(cls, obj: Any) -> bool:
        return isinstance(obj, (CatBoostClassifier, CatBoostRegressor))

    @classmethod
    def process(cls, obj: Any, **kwargs) -> ModelType:
        methods = {
            "predict": Signature(
                name="predict",
                args=[
                    Argument(key="data", type=UnspecifiedDatasetType())
                ],  # TODO: https://github.com/iterative/mlem/issues/21
                returns=UnspecifiedDatasetType(),
            )
        }
        if isinstance(obj, CatBoostClassifier):
            methods["predict_proba"] = Signature(
                name="predict_proba",
                args=[Argument(key="data", type=UnspecifiedDatasetType())],
                # TODO: https://github.com/iterative/mlem/issues/21
                returns=UnspecifiedDatasetType(),
            )
        return CatBoostModel(model=obj, methods=methods)
