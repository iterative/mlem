from typing import Any, ClassVar, Optional, Type, Union

from fastai.data.transforms import Category
from fastai.learner import Learner, load_learner
from fastai.vision.core import PILImage
from pydantic import BaseModel

from mlem.core.artifacts import Artifacts
from mlem.core.dataset_type import DatasetHook, DatasetSerializer, DatasetType
from mlem.core.hooks import IsInstanceHookMixin
from mlem.core.model import BufferModelIO, ModelHook, ModelType, Signature
from mlem.core.requirements import Requirements


class FastAIModelIO(BufferModelIO):
    type: ClassVar = "fastai"

    def save_model(self, model: Any, path: str):
        model.export(path)

    def load(self, artifacts: Artifacts):
        with artifacts[self.art_name].open() as f:
            return load_learner(f)


class FastAIModel(ModelType, ModelHook, IsInstanceHookMixin):
    type: ClassVar = "fastai"
    valid_types: ClassVar = (Learner,)
    io: FastAIModelIO = FastAIModelIO()

    @classmethod
    def process(
        cls, obj: "Learner", sample_data: Optional[Any] = None, **kwargs
    ) -> ModelType:

        return FastAIModel(
            methods={
                "predict": Signature.from_method(
                    obj.predict, item=sample_data, auto_infer=True
                )
            }
        )


class CategoryDataType(
    DatasetType, DatasetSerializer, DatasetHook, IsInstanceHookMixin
):
    type: ClassVar = "fastai_category"
    valid_types: ClassVar = (Category,)
    value: str

    def serialize(self, instance: Any) -> dict:
        raise NotImplementedError  # TODO

    def deserialize(self, obj: dict) -> Any:
        raise NotImplementedError  # TODO

    def get_model(self, prefix: str = "") -> Union[Type[BaseModel], type]:
        raise NotImplementedError  # TODO

    def get_requirements(self) -> Requirements:
        return Requirements.new("fastai")

    @classmethod
    def process(cls, obj: Any, **kwargs):
        return CategoryDataType(value=str(obj))

    def get_writer(self, **kwargs):
        raise NotImplementedError  # TODO


class PILImageDataType(
    DatasetType, DatasetSerializer, DatasetHook, IsInstanceHookMixin
):
    type: ClassVar = "fastai_pil_image"
    valid_types: ClassVar = (PILImage,)

    def serialize(self, instance: Any) -> dict:
        raise NotImplementedError  # TODO

    def deserialize(self, obj: dict) -> Any:
        raise NotImplementedError  # TODO

    def get_model(self, prefix: str = "") -> Union[Type[BaseModel], type]:
        raise NotImplementedError  # TODO

    def get_requirements(self) -> Requirements:
        return Requirements.new("fastai")

    @classmethod
    def process(cls, obj: Any, **kwargs):
        return PILImageDataType()

    def get_writer(self, **kwargs):
        raise NotImplementedError  # TODO
