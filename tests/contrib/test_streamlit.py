from typing import List

from pydantic import BaseModel

from mlem.contrib.streamlit.server import StreamlitServer
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


def test_custom_template(tmpdir):
    template_path = str(tmpdir / "template")
    with open(template_path, "w", encoding="utf8") as f:
        f.write(
            """{{page_title}}
{{title}}
{{description}}
{{server_host}}
{{server_port}}
{{custom_arg}}"""
        )
    server = StreamlitServer(
        template=template_path,
        page_title="page title",
        title="title",
        description="description",
        server_host="host",
        server_port=0,
        args={"custom_arg": "custom arg"},
    )
    path = str(tmpdir / "script")
    server._write_streamlit_script(path)  # pylint: disable=protected-access

    with open(path, encoding="utf8") as f:
        assert (
            f.read()
            == """page title
title
description
host
0
custom arg"""
        )
