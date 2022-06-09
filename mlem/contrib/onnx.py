from typing import IO, Any, ClassVar, Optional, Text, Union

import numpy as np
import onnx
import onnxruntime as onnxrt
import pandas as pd
from numpy.typing import DTypeLike
from onnx import ModelProto, ValueInfoProto, load_model
from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE

from mlem.core.hooks import IsInstanceHookMixin
from mlem.core.model import (
    ModelHook,
    ModelIO,
    ModelProtoIO,
    ModelType,
    Signature,
)
from mlem.core.requirements import InstallableRequirement, Requirements


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


class ONNXWrappedModel:
    """
    Wrapper for `onnx` models and onnx runtime.
    """

    def __init__(self, model: Union[ModelProto, IO[bytes], Text]):
        self.model = (
            model if isinstance(model, ModelProto) else load_model(model)
        )
        self.runtime_session = self._create_runtime_session()

    def _create_runtime_session(self):
        """Creates onnx runtime inference session"""
        # TODO - add support for runtime providers, options. add support for GPU devices.
        return onnxrt.InferenceSession(self.model.SerializeToString())

    def predict(self, data: Any) -> Any:
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
        for input_data in pred_onnx:
            if isinstance(
                input_data, list
            ):  # TODO - temporary workaround to fix fastapi model issues
                output.append(pd.DataFrame(input_data).to_numpy())
            else:
                output.append(input_data)

        return output


class ONNXModel(ModelType, ModelHook, IsInstanceHookMixin):
    """
    :class:`mlem.core.model.ModelType` implementation for `onnx` models
    """

    type: ClassVar[str] = "onnx"
    io: ModelIO = ModelProtoIO()
    valid_types: ClassVar = (ModelProto,)

    @classmethod
    def process(
        cls, obj: Any, sample_data: Optional[Any] = None, **kwargs
    ) -> ModelType:
        obj = ONNXWrappedModel(obj)
        # TODO - use ONNX infer shapes.
        onnxrt_predict = Signature.from_method(
            obj.predict, auto_infer=sample_data is not None, data=sample_data
        )
        methods = {
            "predict": onnxrt_predict,
        }

        return ONNXModel(io=ModelProtoIO(), methods=methods).bind(obj)

    def get_requirements(self) -> Requirements:
        return super().get_requirements() + InstallableRequirement.from_module(
            onnx
        )
