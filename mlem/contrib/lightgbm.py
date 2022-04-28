import os
import posixpath
import tempfile
from typing import Any, ClassVar, Optional, Tuple, Type

import lightgbm as lgb
from pydantic import BaseModel

from mlem.constants import PREDICT_METHOD_NAME
from mlem.core.artifacts import Artifacts, Storage
from mlem.core.dataset_type import (
    DatasetHook,
    DatasetReader,
    DatasetSerializer,
    DatasetType,
    DatasetWriter,
)
from mlem.core.errors import DeserializationError, SerializationError
from mlem.core.hooks import IsInstanceHookMixin
from mlem.core.model import ModelHook, ModelIO, ModelType, Signature
from mlem.core.requirements import (
    AddRequirementHook,
    InstallableRequirement,
    Requirement,
    Requirements,
    UnixPackageRequirement,
)

LGB_REQUIREMENT = UnixPackageRequirement(package_name="libgomp1")


class LightGBMDatasetType(
    DatasetType, DatasetSerializer, DatasetHook, IsInstanceHookMixin
):
    """
    :class:`.DatasetType` implementation for `lightgbm.Dataset` type

    :param inner: :class:`.DatasetType` instance for underlying data
    """

    type: ClassVar[str] = "lightgbm"
    valid_types: ClassVar = (lgb.Dataset,)
    inner: DatasetType

    def serialize(self, instance: Any) -> dict:
        self.check_type(instance, lgb.Dataset, SerializationError)
        return self.inner.get_serializer().serialize(instance.data)

    def deserialize(self, obj: dict) -> Any:
        v = self.inner.get_serializer().deserialize(obj)
        try:
            return lgb.Dataset(v, free_raw_data=False)
        except ValueError as e:
            raise DeserializationError(
                f"object: {obj} could not be converted to lightgbm dataset"
            ) from e

    def get_requirements(self) -> Requirements:
        return (
            Requirements.new([InstallableRequirement.from_module(lgb)])
            + self.inner.get_requirements()
            + LGB_REQUIREMENT
        )

    def get_reader(self, **kwargs) -> DatasetReader:
        return LightGBMDatasetReader()

    def get_writer(self, **kwargs) -> DatasetWriter:
        return LightGBMDatasetWriter()

    @classmethod
    def process(cls, obj: Any, **kwargs) -> DatasetType:
        return LightGBMDatasetType(inner=DatasetType.create(obj.data))

    def get_model(self) -> Type[BaseModel]:
        return self.inner.get_serializer().get_model()


DATA_FILE = "data.npy"


class LightGBMDatasetWriter(DatasetWriter):
    type: ClassVar[str] = "lightgbm"

    def write(
        self, dataset: DatasetType, storage: Storage, path: str
    ) -> Tuple[DatasetReader, Artifacts]:
        _, art = dataset.inner.get_writer().write(dataset.inner, storage, path)  # type: ignore
        return LightGBMDatasetReader(dataset_type=dataset), art


class LightGBMDatasetReader(DatasetReader):
    type: ClassVar[str] = "lightgbm"
    dataset_type: LightGBMDatasetType

    def read(self, artifacts: Artifacts) -> DatasetType:
        if len(artifacts) != 1:
            raise ValueError(
                f"Wrong artifacts {artifacts}: should be one {DATA_FILE} file"
            )
        data = self.dataset_type.inner.get_reader(
            dataset_type=self.dataset_type.inner
        ).read(artifacts)
        self.dataset_type.inner = self.dataset_type.inner.copy().bind(data)
        return self.dataset_type


class LightGBMModelIO(ModelIO):
    """
    :class:`.ModelIO` implementation for `lightgbm.Booster` type
    """

    type: ClassVar[str] = "lightgbm_io"
    model_file_name = "model.lgb"

    def dump(self, storage: Storage, path, model) -> Artifacts:
        with tempfile.TemporaryDirectory(prefix="mlem_lightgbm_dump") as f:
            model_path = os.path.join(f, self.model_file_name)
            model.save_model(model_path)
            fs_path = posixpath.join(path, self.model_file_name)
            return {self.art_name: storage.upload(model_path, fs_path)}

    def load(self, artifacts: Artifacts):
        if len(artifacts) != 1:
            raise ValueError(
                f"Invalid artifacts: should be one {self.model_file_name} file"
            )

        with tempfile.TemporaryDirectory(
            prefix="mlem_lightgbm_load"
        ) as tmpdir:
            local_path = os.path.join(tmpdir, self.model_file_name)
            artifacts[self.art_name].materialize(
                local_path,
            )
            return lgb.Booster(model_file=local_path)


class LightGBMModel(ModelType, ModelHook, IsInstanceHookMixin):
    """
    :class:`.ModelType` implementation for `lightgbm.Booster` type
    """

    type: ClassVar[str] = "lightgbm"
    valid_types: ClassVar = (lgb.Booster,)
    io: ModelIO = LightGBMModelIO()

    @classmethod
    def process(
        cls, obj: Any, sample_data: Optional[Any] = None, **kwargs
    ) -> ModelType:
        gbm_model = LightGBMModel(model=obj, methods={})
        gbm_model.methods = {
            PREDICT_METHOD_NAME: Signature.from_method(
                gbm_model.predict,
                auto_infer=sample_data is not None,
                data=sample_data,
            ),
            "lightgbm_predict": Signature.from_method(
                obj.predict, auto_infer=sample_data is None, data=sample_data
            ),
        }
        return gbm_model

    def predict(self, data):
        if isinstance(data, lgb.Dataset):
            data = data.data
        return self.model.predict(data)

    def get_requirements(self) -> Requirements:
        return (
            super().get_requirements()
            + InstallableRequirement.from_module(mod=lgb)
            + LGB_REQUIREMENT
        )


class LGBMLibgompHook(AddRequirementHook):
    to_add = LGB_REQUIREMENT

    @classmethod
    def is_object_valid(cls, obj: Requirement) -> bool:
        return (
            isinstance(obj, InstallableRequirement)
            and obj.module == "lightgbm"
        )
