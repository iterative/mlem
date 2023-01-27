from typing import List

from pydantic import BaseModel

from mlem.contrib.streamlit.utils import augment_model


def test_augment_model():
    class M1(BaseModel):
        field: str

    aug, model = augment_model(M1)
    assert model == M1
    assert aug(1) == 1

    class M2(BaseModel):
        field: List[str]

    aug, model = augment_model(M2)
    assert model == str
    assert aug("1") == M2(field=["1"])

    class M3(BaseModel):
        field: List[M1]

    aug, model = augment_model(M3)
    assert model == M1
    assert aug(M1(field="1")) == M3(field=[M1(field="1")])

    class M4(BaseModel):
        field: List[str]
        field2: List[str]

    aug, model = augment_model(M4)
    assert model is None
