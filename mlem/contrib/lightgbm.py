import os
import posixpath
import tempfile
from typing import Any, ClassVar, Iterator, List, Optional, Tuple, Type

import lightgbm as lgb
from pydantic import BaseModel

from mlem.constants import PREDICT_METHOD_NAME
from mlem.core.artifacts import Artifacts, Storage
from mlem.core.data_type import (
    DataAnalyzer,
    DataHook,
    DataReader,
    DataSerializer,
    DataType,
    DataWriter,
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


class LightGBMDataType(
    DataType, DataSerializer, DataHook, IsInstanceHookMixin
):
    """
    :class:`.DataType` implementation for `lightgbm.Dataset` type

    :param inner: :class:`.DataType` instance for underlying data
    """

    type: ClassVar[str] = "lightgbm"
    valid_types: ClassVar = (lgb.Dataset,)
    inner: DataType

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

    def get_writer(
        self, project: str = None, filename: str = None, **kwargs
    ) -> DataWriter:
        return LightGBMDataWriter(**kwargs)

    @classmethod
    def process(cls, obj: Any, **kwargs) -> DataType:
        return LightGBMDataType(inner=DataAnalyzer.analyze(obj.data))

    def get_model(self, prefix: str = "") -> Type[BaseModel]:
        return self.inner.get_serializer().get_model(prefix)


class LightGBMDataWriter(DataWriter):
    type: ClassVar[str] = "lightgbm"

    def write(
        self, data: DataType, storage: Storage, path: str
    ) -> Tuple[DataReader, Artifacts]:
        if not isinstance(data, LightGBMDataType):
            raise ValueError(
                f"expected data to be of LightGBMDataType, got {type(data)} instead"
            )
        lightgbm_construct = data.data.construct()
        raw_data = lightgbm_construct.get_data()
        underlying_labels = lightgbm_construct.get_label().tolist()
        inner_reader, art = data.inner.get_writer().write(
            data.inner.copy().bind(raw_data), storage, path
        )
        return (
            LightGBMDataReader(
                data_type=data,
                inner=inner_reader,
                label=underlying_labels,
            ),
            art,
        )


class LightGBMDataReader(DataReader):
    type: ClassVar[str] = "lightgbm"
    data_type: LightGBMDataType
    inner: DataReader
    label: List

    def read(self, artifacts: Artifacts) -> DataType:
        inner_data_type = self.inner.read(artifacts)
        return LightGBMDataType(inner=inner_data_type).bind(
            lgb.Dataset(
                inner_data_type.data, label=self.label, free_raw_data=False
            )
        )

    def read_batch(
        self, artifacts: Artifacts, batch_size: int
    ) -> Iterator[DataType]:
        raise NotImplementedError


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
