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

    def serialize(self, data_type: NumpyNdarrayType, instance: Any) -> bytes:
        im = Image.fromarray(instance)
        return im.tobytes()

    @contextlib.contextmanager
    def dump(
        self, data_type: NumpyNdarrayType, instance: Any
    ) -> Iterator[BinaryIO]:
        buffer = BytesIO()
        Image.fromarray(instance).save(buffer)
        buffer.seek(0)
        yield buffer

    def deserialize(
        self, data_type: NumpyNdarrayType, obj: Union[bytes, BinaryIO]
    ) -> Any:
        if isinstance(obj, bytes):
            obj = BytesIO(obj)
        im = Image.open(obj)
        return numpy.array(im)
