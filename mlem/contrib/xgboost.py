import os
import tempfile
from typing import Any, ClassVar, Dict, List, Optional

import xgboost
from fsspec import AbstractFileSystem

from mlem.contrib.numpy import python_type_from_np_string_repr
from mlem.core.artifacts import Artifacts
from mlem.core.dataset_type import (
    DatasetHook,
    DatasetType,
    DatasetWriter,
    UnspecifiedDatasetType,
)
from mlem.core.errors import DeserializationError, SerializationError
from mlem.core.hooks import IsInstanceHookMixin
from mlem.core.model import Argument, ModelHook, ModelIO, ModelType, Signature
from mlem.core.requirements import (
    InstallableRequirement,
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

    type: ClassVar = "xgboost_dmatrix"
    types: ClassVar = (xgboost.DMatrix,)

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
        except (ValueError, TypeError):
            raise DeserializationError(
                f"given object: {obj} could not be converted to xgboost matrix"
            )

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

    def get_writer(self, **kwargs) -> "DatasetWriter":
        raise NotImplementedError()  # TODO: https://github.com/iterative/mlem/issues/35


class XGBoostModelIO(ModelIO):
    """
    :class:`~.ModelIO` implementation for XGBoost models
    """

    type: ClassVar = "xgboost_io"
    model_file_name = "model.xgb"

    def dump(
        self, fs: AbstractFileSystem, path, model: xgboost.Booster
    ) -> Artifacts:
        with tempfile.TemporaryDirectory(prefix="mlem_xgboost_dump") as f:
            local_path = os.path.join(f, self.model_file_name)
            model.save_model(local_path)
            remote_path = os.path.join(path, self.model_file_name)
            fs.upload(local_path, remote_path)
            return [remote_path]

    def load(self, fs: AbstractFileSystem, path):
        model = xgboost.Booster()
        with tempfile.TemporaryDirectory(prefix="mlem_xgboost_load") as f:
            rpath = os.path.join(path, self.model_file_name)
            lpath = os.path.join(f, self.model_file_name)
            fs.download(rpath, lpath)
            model.load_model(lpath)
        return model


class XGBoostModel(
    XGBoostRequirement, ModelType, ModelHook, IsInstanceHookMixin
):
    """
    :class:`~.ModelType` implementation for XGBoost models
    """

    type: ClassVar = "xgboost"
    types: ClassVar = (xgboost.Booster,)

    io: ModelIO = XGBoostModelIO()

    @classmethod
    def process(cls, obj: Any, **kwargs) -> ModelType:
        methods = {
            "predict": Signature(
                name="_predict",
                args=[Argument(key="data", type=UnspecifiedDatasetType())],
                returns=UnspecifiedDatasetType(),  # TODO: https://github.com/iterative/mlem/issues/21
            )
        }
        return XGBoostModel(model=obj, methods=methods)

    def _predict(self, data):
        if not isinstance(data, xgboost.DMatrix):
            data = xgboost.DMatrix(data)
        return self.model.predict(data)
