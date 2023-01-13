"""PIL Image support
Extension type: data

Serializer for PIL Images to/from numpy arrays
"""
import contextlib
from io import BytesIO
from typing import Any, BinaryIO, ClassVar, Iterator, Union

import numpy
from PIL import Image

from mlem.contrib.numpy import NumpyNdarrayType
from mlem.core.data_type import BinarySerializer


class PILImageSerializer(BinarySerializer):
    """Serializes numpy arrays to/from images"""

    type: ClassVar = "pil_numpy"
    support_files: ClassVar = True

    format: str = "jpeg"
    "Image format to use"

    def serialize(self, data_type: NumpyNdarrayType, instance: Any) -> bytes:
        with self.dump(data_type, instance) as b:
            return b.getvalue()

    @contextlib.contextmanager
    def dump(
        self, data_type: NumpyNdarrayType, instance: Any
    ) -> Iterator[BytesIO]:
        buffer = BytesIO()
        Image.fromarray(instance).save(buffer, format=self.format)
        buffer.seek(0)
        yield buffer

    def deserialize(
        self, data_type: NumpyNdarrayType, obj: Union[bytes, BinaryIO]
    ) -> Any:
        if isinstance(obj, bytes):
            obj = BytesIO(obj)
        im = Image.open(obj)
        return numpy.array(im)
