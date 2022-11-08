"""LightGBM models support
Extension type: model

ModelType and ModelIO implementations for `lightgbm.Booster` as well as
LightGBMDataType with Reader and Writer for `lightgbm.Dataset`
"""
import os
import posixpath
import tempfile
from typing import Any, ClassVar, Iterator, Optional, Tuple, Type

import flatdict
import lightgbm as lgb
from pydantic import BaseModel

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
LIGHTGBM_DATA = "inner"
LIGHTGBM_LABEL = "label"


class LightGBMDataType(
    DataType, DataSerializer, DataHook, IsInstanceHookMixin
):
    """
    :class:`.DataType` implementation for `lightgbm.Dataset` type

    :param inner: :class:`.DataType` instance for underlying data
    :param labels: :class:`.DataType` instance for underlying labels
    """

    type: ClassVar[str] = "lightgbm"
    valid_types: ClassVar = (lgb.Dataset,)
    inner: DataType
    """DataType of Inner"""
    labels: Optional[DataType]
    """DataType of Labels"""

    def serialize(self, instance: Any) -> dict:
        self.check_type(instance, lgb.Dataset, SerializationError)
        if self.labels is not None:
            return {
                LIGHTGBM_DATA: self.inner.get_serializer().serialize(
                    instance.data
                ),
                LIGHTGBM_LABEL: self.labels.get_serializer().serialize(
                    instance.label
                ),
            }
        return self.inner.get_serializer().serialize(instance.data)

    def deserialize(self, obj: dict) -> Any:
        if self.labels is not None:
            data = self.inner.get_serializer().deserialize(obj[LIGHTGBM_DATA])
            label = self.labels.get_serializer().deserialize(
                obj[LIGHTGBM_LABEL]
            )
        else:
            data = self.inner.get_serializer().deserialize(obj)
            label = None
        try:
            return lgb.Dataset(data, label=label, free_raw_data=False)
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
        return LightGBMDataType(
            inner=DataAnalyzer.analyze(obj.data),
            labels=DataAnalyzer.analyze(obj.label)
            if obj.label is not None
            else None,
        )

    def get_model(self, prefix: str = "") -> Type[BaseModel]:
        return self.inner.get_serializer().get_model(prefix)


class LightGBMDataWriter(DataWriter):
    """Wrapper writer for lightgbm.Dataset objects"""

    type: ClassVar[str] = "lightgbm"

    def write(
        self, data: DataType, storage: Storage, path: str
    ) -> Tuple[DataReader, Artifacts]:
        if not isinstance(data, LightGBMDataType):
            raise ValueError(
                f"expected data to be of LightGBMDataType, got {type(data)} instead"
            )

        lightgbm_raw = data.data

        if data.labels is not None:
            inner_reader, inner_art = data.inner.get_writer().write(
                data.inner.copy().bind(lightgbm_raw.data),
                storage,
                posixpath.join(path, LIGHTGBM_DATA),
            )
            labels_reader, labels_art = data.labels.get_writer().write(
                data.labels.copy().bind(lightgbm_raw.label),
                storage,
                posixpath.join(path, LIGHTGBM_LABEL),
            )
            res = dict(
                flatdict.FlatterDict(
                    {LIGHTGBM_DATA: inner_art, LIGHTGBM_LABEL: labels_art},
                    delimiter="/",
                )
            )
        else:
            inner_reader, inner_art = data.inner.get_writer().write(
                data.inner.copy().bind(lightgbm_raw.data),
                storage,
                path,
            )
            res = inner_art
            labels_reader = None

        return (
            LightGBMDataReader(
                data_type=data,
                inner=inner_reader,
                labels=labels_reader,
            ),
            res,
        )


class LightGBMDataReader(DataReader):
    """Wrapper reader for lightgbm.Dataset objects"""

    type: ClassVar[str] = "lightgbm"
    data_type: LightGBMDataType
    inner: DataReader
    """DataReader of Inner"""
    labels: Optional[DataReader]
    """DataReader of Labels"""

    def read(self, artifacts: Artifacts) -> DataType:
        if self.labels is not None:
            artifacts = flatdict.FlatterDict(artifacts, delimiter="/")
            inner_data_type = self.inner.read(artifacts[LIGHTGBM_DATA])  # type: ignore[arg-type]
            labels_data_type = self.labels.read(artifacts[LIGHTGBM_LABEL])  # type: ignore[arg-type]
        else:
            inner_data_type = self.inner.read(artifacts)
            labels_data_type = None
        return LightGBMDataType(
            inner=inner_data_type, labels=labels_data_type
        ).bind(
            lgb.Dataset(
                inner_data_type.data,
                label=labels_data_type.data
                if labels_data_type is not None
                else None,
                free_raw_data=False,
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
    model_file_name: str = "model.lgb"
    """Filename to use"""

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
    """LightGBMModelIO"""

    @classmethod
    def process(
        cls, obj: Any, sample_data: Optional[Any] = None, **kwargs
    ) -> ModelType:
        og_data = sample_data
        if sample_data is not None and isinstance(sample_data, lgb.Dataset):
            sample_data = sample_data.data

        signature = Signature.from_method(
            obj.predict, auto_infer=sample_data is not None, data=sample_data
        )
        if og_data is not None:
            signature.args[0].type_ = DataAnalyzer.analyze(og_data)

        return LightGBMModel(
            model=obj,
            methods={
                "predict": signature,
            },
        )

    def predict(self, data, **kwargs):
        if isinstance(data, lgb.Dataset):
            data = data.data
        return self.model.predict(data, **kwargs)

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
