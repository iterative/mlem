"""Torch Image Serializer
Extension type: serving

TorchImageSerializer implementation
"""
import contextlib
from io import BytesIO
from typing import Any, BinaryIO, ClassVar, Iterator, Union

from torch import frombuffer, uint8
from torchvision.io import decode_image
from torchvision.transforms import ToPILImage

from mlem.contrib.torch import TorchTensorDataType
from mlem.core.data_type import BinarySerializer


def _to_buffer(instance):
    buffer = BytesIO()
    ToPILImage()(instance).save(buffer, "JPEG")
    buffer.seek(0)
    return buffer


class TorchImageSerializer(BinarySerializer):
    """Serializes torch tensors to/from images"""

    type: ClassVar = "torch_image"
    support_files: ClassVar = True

    def serialize(
        self, data_type: TorchTensorDataType, instance: Any
    ) -> bytes:
        return _to_buffer(instance).read()

    @contextlib.contextmanager
    def dump(
        self, data_type: TorchTensorDataType, instance: Any
    ) -> Iterator[BinaryIO]:
        yield _to_buffer(instance)

    def deserialize(
        self, data_type: TorchTensorDataType, obj: Union[bytes, BinaryIO]
    ) -> Any:
        if isinstance(obj, bytes):
            buffer = obj
        else:
            buffer = obj.read()
        return decode_image(frombuffer(buffer, dtype=uint8))
