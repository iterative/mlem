"""ONNX models support
Extension type: model

ModelType and ModelIO implementations for `onnx.ModelProto`
"""
from typing import Any, ClassVar, List, Optional, Union

import numpy as np
import onnx
import onnxruntime as onnxrt
import pandas as pd
from numpy.typing import DTypeLike
from onnx import ModelProto, ValueInfoProto
from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE

from mlem.core.artifacts import Artifacts, Storage
from mlem.core.hooks import IsInstanceHookMixin
from mlem.core.model import ModelHook, ModelIO, ModelType, Signature
from mlem.core.requirements import InstallableRequirement, Requirements
from mlem.utils.backport import cached_property
from mlem.utils.module import get_object_requirements


def convert_to_numpy(
    data: Union[np.ndarray, pd.DataFrame], dtype: DTypeLike
) -> np.ndarray:
    """Converts input data to numpy"""
    if isinstance(data, np.ndarray):
        pass
    elif isinstance(data, pd.DataFrame):
        data = data.to_numpy()
    else:
        raise TypeError(f"input data type: {type(data)} is not supported")
    return data.astype(dtype=dtype)


def get_onnx_to_numpy_type(value_info: ValueInfoProto) -> DTypeLike:
    """Returns numpy equivalent type of onnx value info"""
    onnx_type = value_info.type.tensor_type.elem_type
    return TENSOR_TYPE_TO_NP_TYPE[onnx_type]


class ModelProtoIO(ModelIO):
    """IO for ONNX model object"""

    type: ClassVar[str] = "model_proto"

    def dump(self, storage: Storage, path: str, model) -> Artifacts:
        path = f"{path}/model.onnx"
        with storage.open(path) as (f, art):
            onnx.save_model(
                model,
                f,
                save_as_external_data=True,
                location="tensors",
                size_threshold=0,
                all_tensors_to_one_file=True,
            )
        return {self.art_name: art}

    def load(self, artifacts: Artifacts):
        if len(artifacts) != 1:
            raise ValueError("Invalid artifacts: should be one .onnx file")
        with artifacts[self.art_name].open() as f:
            return onnx.load_model(f)


class ONNXModel(ModelType, ModelHook, IsInstanceHookMixin):
    """
    :class:`mlem.core.model.ModelType` implementation for `onnx` models
    """

    type: ClassVar[str] = "onnx"
    io: ModelIO = ModelProtoIO()
    valid_types: ClassVar = (ModelProto,)

    class Config:
        keep_untouched = (cached_property,)

    @classmethod
    def process(
        cls, obj: Any, sample_data: Optional[Any] = None, **kwargs
    ) -> ModelType:

        model = ONNXModel(io=ModelProtoIO(), methods={}).bind(obj)
        # TODO - use ONNX infer shapes.
        onnxrt_predict = Signature.from_method(
            model.predict, auto_infer=sample_data is not None, data=sample_data
        )
        model.methods = {
            "predict": onnxrt_predict,
        }

        return model

    @cached_property
    def runtime_session(self) -> onnxrt.InferenceSession:
        """Provides onnx runtime inference session"""
        # TODO - add support for runtime providers, options. add support for GPU devices.
        return onnxrt.InferenceSession(self.model.SerializeToString())

    def predict(self, data: Union[List, np.ndarray, pd.DataFrame]) -> Any:
        """Returns inference output for given input data"""
        model_inputs = self.runtime_session.get_inputs()

        if not isinstance(data, list):
            data = [data]

        if len(model_inputs) != len(data):
            raise ValueError(
                f"no of inputs provided: {len(data)}, "
                f"expected: {len(model_inputs)}"
            )

        input_dict = {}
        for model_input, input_data in zip(self.model.graph.input, data):
            input_dict[model_input.name] = convert_to_numpy(
                input_data, get_onnx_to_numpy_type(model_input)
            )

        label_names = [out.name for out in self.runtime_session.get_outputs()]
        pred_onnx = self.runtime_session.run(label_names, input_dict)

        output = []
        for output_data in pred_onnx:
            if isinstance(
                output_data, list
            ):  # TODO - temporary workaround to fix fastapi model issues
                output.append(pd.DataFrame(output_data).to_numpy())
            else:
                output.append(output_data)

        return output

    def get_requirements(self) -> Requirements:
        return (
            super().get_requirements()
            + InstallableRequirement.from_module(onnx)
            + get_object_requirements(self.predict)
            + Requirements.new(
                InstallableRequirement(module="protobuf", version="3.20.1")
            )
        )
        # https://github.com/protocolbuffers/protobuf/issues/10051
