import matplotlib.image
import numpy as np
import pytest

from mlem.contrib.numpy import NumpyNdarrayType
from mlem.contrib.pil import PILImageSerializer
from mlem.core.data_type import DataType
from tests.conftest import resource_path

IMAGE_PATH = resource_path(__file__, "im.jpg")


@pytest.fixture
def np_image():
    return matplotlib.image.imread(resource_path(__file__, "im.jpg"))


def test_pil_serializer(np_image):
    data_type = DataType.create(np_image)

    assert isinstance(data_type, NumpyNdarrayType)

    payload = PILImageSerializer().serialize(data_type, np_image)
    assert isinstance(payload, bytes)
    image_again = PILImageSerializer().deserialize(data_type, payload)

    assert np.equal(image_again, np_image)
