import os
import posixpath
import tempfile
from typing import Any, ClassVar, Dict, List, Optional

import xgboost

from mlem.constants import PREDICT_METHOD_NAME
from mlem.contrib.numpy import python_type_from_np_string_repr
from mlem.core.artifacts import Artifacts, Storage
from mlem.core.dataset_type import DatasetHook, DatasetType, DatasetWriter
from mlem.core.errors import DeserializationError, SerializationError
from mlem.core.hooks import IsInstanceHookMixin
from mlem.core.model import ModelHook, ModelIO, ModelType, Signature
from mlem.core.requirements import (
    AddRequirementHook,
    InstallableRequirement,
    Requirement,
    Requirements,
    UnixPackageRequirement,
    WithRequirements,
)

XGB_REQUIREMENT = UnixPackageRequirement(package_name="libgomp1")


class XGBoostRequirement(WithRequirements):
    def get_requirements(self) -> Requirements:
        return (
            Requirements.new([InstallableRequirement.from_module(xgboost)])
            + XGB_REQUIREMENT
        )


class DMatrixDatasetType(
    XGBoostRequirement,
    DatasetType,
    DatasetHook,
    IsInstanceHookMixin,
):
    """
    :class:`~.DatasetType` implementation for xgboost.DMatrix type

    :param is_from_list: whether DMatrix can be constructed from list
    :param feature_type_names: string representation of feature types
    :param feature_names: list of feature names
    """

    type: ClassVar[str] = "xgboost_dmatrix"
    valid_types: ClassVar = (xgboost.DMatrix,)

    is_from_list: bool
    feature_type_names: Optional[List[str]]
    feature_names: Optional[List[str]] = None

    @property
    def feature_types(self):
        return (
            [
                python_type_from_np_string_repr(t)
                for t in self.feature_type_names
            ]
            if self.feature_type_names
            else ["float32" for _ in range(len(self.feature_names))]
        )

    def serialize(self, instance: xgboost.DMatrix) -> Dict[Any, Any]:
        """
        Raises an error because there is no way to extract original data from DMatrix
        """
        raise SerializationError(
            "xgboost matrix does not support serialization"
        )

    def deserialize(self, obj: Dict[Any, Any]) -> xgboost.DMatrix:
        try:
            return xgboost.DMatrix(obj)
        except (ValueError, TypeError) as e:
            raise DeserializationError(
                f"given object: {obj} could not be converted to xgboost matrix"
            ) from e

    @classmethod
    def from_dmatrix(cls, dmatrix: xgboost.DMatrix):
        """
        Factory method to extract :class:`~.DatasetType` from actual xgboost.DMatrix

        :param dmatrix: obj to create :class:`~.DatasetType` from
        :return: :class:`DMatrixDatasetType`
        """
        is_from_list = (
            dmatrix.feature_names is None
        )  # (dmatrix.feature_names == [f'f{i}' for i in range(dmatrix.num_col())])
        return DMatrixDatasetType(
            is_from_list=is_from_list,
            feature_type_names=dmatrix.feature_types,
            feature_names=dmatrix.feature_names,
        )

    @classmethod
    def process(cls, obj: xgboost.DMatrix, **kwargs) -> DatasetType:
        return DMatrixDatasetType.from_dmatrix(obj)

    def get_writer(self, **kwargs) -> DatasetWriter:
        raise NotImplementedError()  # TODO: https://github.com/iterative/mlem/issues/35


class XGBoostModelIO(ModelIO):
    """
    :class:`~.ModelIO` implementation for XGBoost models
    """

    type: ClassVar[str] = "xgboost_io"
    model_file_name = "model.xgb"

    def dump(
        self, storage: Storage, path, model: xgboost.Booster
    ) -> Artifacts:
        with tempfile.TemporaryDirectory(prefix="mlem_xgboost_dump") as f:
            local_path = os.path.join(f, self.model_file_name)
            model.save_model(local_path)
            remote_path = posixpath.join(path, self.model_file_name)
            return [storage.upload(local_path, remote_path)]

    def load(self, artifacts: Artifacts):
        if len(artifacts) != 1:
            raise ValueError(
                f"Invalid artifacts: should be one {self.model_file_name} file"
            )

        model = xgboost.Booster()
        with tempfile.TemporaryDirectory(prefix="mlem_xgboost_load") as f:
            lpath = os.path.join(f, self.model_file_name)
            artifacts[0].materialize(lpath)
            model.load_model(lpath)
        return model


class XGBoostModel(ModelType, ModelHook, IsInstanceHookMixin):
    """
    :class:`~.ModelType` implementation for XGBoost models
    """

    type: ClassVar[str] = "xgboost"
    valid_types: ClassVar = (xgboost.Booster,)

    io: ModelIO = XGBoostModelIO()

    @classmethod
    def process(
        cls, obj: Any, sample_data: Optional[Any] = None, **kwargs
    ) -> ModelType:
        model = XGBoostModel(model=obj, methods={})
        methods = {
            PREDICT_METHOD_NAME: Signature.from_method(
                model.predict,
                auto_infer=sample_data is not None,
                data=sample_data,
            ),
            "xgboost_predict": Signature.from_method(
                obj.predict, auto_infer=sample_data is None, data=sample_data
            ),
        }
        model.methods = methods
        return model

    def predict(self, data):
        if not isinstance(data, xgboost.DMatrix):
            data = xgboost.DMatrix(data)
        return self.model.predict(data)

    def get_requirements(self) -> Requirements:
        return (
            super().get_requirements()
            + InstallableRequirement.from_module(xgboost)
            + XGB_REQUIREMENT
        )


class XGBLibgopmHook(AddRequirementHook):
    to_add = XGB_REQUIREMENT

    @classmethod
    def is_object_valid(cls, obj: Requirement) -> bool:
        return (
            isinstance(obj, InstallableRequirement) and obj.module == "xgboost"
        )
