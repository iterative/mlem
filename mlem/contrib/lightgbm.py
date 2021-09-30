import os
import tempfile
from typing import Any, ClassVar

import lightgbm as lgb
from fsspec import AbstractFileSystem

from mlem.core.artifacts import Artifacts
from mlem.core.dataset_type import (
    DatasetAnalyzer,
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
)

LGB_REQUIREMENT = UnixPackageRequirement(package_name="libgomp1")


class LightGBMDatasetType(DatasetType, DatasetHook, IsInstanceHookMixin):
    """
    :class:`.DatasetType` implementation for `lightgbm.Dataset` type

    :param inner: :class:`.DatasetType` instance for underlying data
    """

    type: ClassVar = "lightgbm"
    types: ClassVar = (lgb.Dataset,)
    inner: DatasetType

    def serialize(self, instance: Any) -> dict:
        self._check_type(instance, lgb.Dataset, SerializationError)
        return self.inner.serialize(instance.data)

    def deserialize(self, obj: dict) -> Any:
        v = self.inner.deserialize(obj)
        try:
            return lgb.Dataset(v, free_raw_data=False)
        except ValueError:
            raise DeserializationError(
                f"object: {obj} could not be converted to lightgbm dataset"
            )

    def get_requirements(self) -> Requirements:
        return (
            Requirements.new([InstallableRequirement.from_module(lgb)])
            + self.inner.get_requirements()
            + LGB_REQUIREMENT
        )

    def get_writer(self, **kwargs) -> "DatasetWriter":
        raise NotImplementedError()

    @classmethod
    def process(cls, obj: Any, **kwargs) -> DatasetType:
        return LightGBMDatasetType(inner=DatasetAnalyzer.analyze(obj.data))


class LightGBMModelIO(ModelIO):
    """
    :class:`.ModelIO` implementation for `lightgbm.Booster` type
    """

    type: ClassVar = "lightgbm_io"
    model_file_name = "model.lgb"

    def dump(self, fs: AbstractFileSystem, path, model) -> Artifacts:
        with tempfile.TemporaryDirectory(prefix="mlem_lightgbm_dump") as f:
            model_path = os.path.join(f, self.model_file_name)
            model.save_model(model_path)
            fs_path = os.path.join(path, self.model_file_name)
            fs.upload(model_path, fs_path)
            return [fs_path]

    def load(self, fs: AbstractFileSystem, path):
        model_file = os.path.join(path, self.model_file_name)
        with tempfile.TemporaryDirectory(
            prefix="mlem_lightgbm_load"
        ) as tmpdir:
            local_path = os.path.join(tmpdir, self.model_file_name)
            fs.download(model_file, local_path)
            return lgb.Booster(model_file=local_path)


class LightGBMModel(ModelType, ModelHook, IsInstanceHookMixin):
    """
    :class:`.ModelType` implementation for `lightgbm.Booster` type
    """

    type: ClassVar = "lightgbm"
    types: ClassVar = (lgb.Booster,)
    io: ModelIO = LightGBMModelIO()

    @classmethod
    def process(cls, obj: Any, **kwargs) -> ModelType:
        return LightGBMModel(
            model=obj,
            methods={
                "predict": Signature(
                    name="_predict",
                    args=[Argument(key="data", type=UnspecifiedDatasetType())],
                    returns=UnspecifiedDatasetType(),  # TODO: https://github.com/iterative/mlem/issues/21
                )
            },
        )

    def _predict(self, data):
        if isinstance(data, lgb.Dataset):
            data = data.data
        return self.model.predict(data)

    def get_requirements(self) -> Requirements:
        return (
            Requirements.new(InstallableRequirement.from_module(mod=lgb))
            + LGB_REQUIREMENT
        )
