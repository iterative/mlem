from typing import ClassVar, List, Optional

from pydantic import BaseModel

from mlem.contrib.docker import DockerImageBuilder
from mlem.contrib.fastapi import FastAPIServer
from mlem.core.base import (
    MlemABC,
    SmartSplitDict,
    build_mlem_object,
    parse_links,
    smart_split,
)
from mlem.core.objects import MlemBuilder, MlemLink, MlemModel, MlemObject
from mlem.runtime.server import Server
from tests.conftest import resource_path


def test_build_model():
    res = build_mlem_object(
        MlemBuilder,
        "docker",
        ["image.name=kek"],
        [f"server={resource_path(__file__, 'server.yaml')}"],
    )
    assert isinstance(res, DockerImageBuilder)
    assert res.image.name == "kek"
    assert isinstance(res.server, FastAPIServer)
    assert res.server.port == 8081


def test_smart_split():
    assert smart_split("a 'b c' d", " ") == ["a", "b c", "d"]
    assert smart_split('a."b.c".d', ".") == ["a", "b.c", "d"]
    assert smart_split("a.'b c'.d", ".") == ["a", "b c", "d"]


def test_smart_split_maxsplit():
    assert smart_split("a=b=c=d", "=", maxsplit=2) == ["a", "b", "c=d"]


def test_parse_links():
    class ModelWithLink(MlemObject):
        field_link: MlemLink
        field: Optional[MlemModel]

    assert parse_links(ModelWithLink, ["field=somepath"]) == (
        [],
        {"field_link": MlemLink(path="somepath", link_type="model")},
    )


class MockMlemABC(MlemABC):
    abs_name: ClassVar = "mock"
    server: Server


def test_build_with_replace():
    res = build_mlem_object(
        MockMlemABC,
        "mock",
        ["server=fastapi", "server.port=8081", "server.host=localhost"],
    )
    assert isinstance(res, MockMlemABC)
    assert isinstance(res.server, FastAPIServer)
    assert res.server.port == 8081
    assert res.server.host == "localhost"

    res = build_mlem_object(
        MockMlemABC,
        "mock",
        ["server=fastapi"],
    )
    assert isinstance(res, MockMlemABC)
    assert isinstance(res.server, FastAPIServer)

    res = build_mlem_object(
        MockMlemABC,
        "mock",
        conf={
            "server": "fastapi",
            "server.port": 8081,
            "server.host": "localhost",
        },
    )
    assert isinstance(res, MockMlemABC)
    assert isinstance(res.server, FastAPIServer)
    assert res.server.port == 8081
    assert res.server.host == "localhost"


def test_build_with_list():
    class MockMlemABCList(MlemABC):
        abs_name: ClassVar = "mock_list"
        values: List[str]

    res = build_mlem_object(
        MockMlemABCList,
        "mock_list",
        ["values.0=a", "values.1=b"],
    )
    assert isinstance(res, MockMlemABCList)
    assert isinstance(res.values, list)
    assert res.values == ["a", "b"]


def test_build_with_list_complex():
    class Value(BaseModel):
        field: str

    class MockMlemABCListComplex(MlemABC):
        abs_name: ClassVar = "mock_list_complex"
        values: List[Value]

    res = build_mlem_object(
        MockMlemABCListComplex,
        "mock_list_complex",
        ["values.0.field=a", "values.1.field=b"],
    )
    assert isinstance(res, MockMlemABCListComplex)
    assert isinstance(res.values, list)
    assert res.values == [Value(field="a"), Value(field="b")]


def test_build_with_list_nested():
    class MockMlemABCListNested(MlemABC):
        abs_name: ClassVar = "mock_list_complex"
        values: List[List[str]]

    res = build_mlem_object(
        MockMlemABCListNested,
        MockMlemABCListNested.abs_name,
        ["values.0.0=a", "values.0.1=b"],
    )
    assert isinstance(res, MockMlemABCListNested)
    assert isinstance(res.values, list)
    assert res.values == [["a", "b"]]


def test_smart_split_dict():
    d = SmartSplitDict(sep=".")
    d["a.b.c"] = 1
    d["a.b.d"] = 2
    d["a.e"] = 3
    d["a.f"] = 4
    d["g"] = 5

    assert d.build() == {"g": 5, "a": {"f": 4, "e": 3, "b": {"d": 2, "c": 1}}}


def test_smart_split_dict_with_list():
    d = SmartSplitDict(sep=".")
    d["a.0"] = 1
    d["a.1"] = 2
    d["b"] = 3

    assert d.build() == {"a": [1, 2], "b": 3}


def test_smart_split_dict_with_nested():
    d = SmartSplitDict(sep=".")
    d["ll.0.0"] = 1
    d["ll.0.1"] = 2
    d["ll.1.0"] = 3
    d["ll.1.1"] = 4
    d["ld.0.a"] = 5
    d["ld.0.b"] = 6
    d["ld.1.a"] = 7
    d["ld.1.b"] = 8
    d["dl.a.0"] = 9
    d["dl.a.1"] = 10
    d["dl.b.0"] = 11
    d["dl.b.1"] = 12
    d["dd.a.a"] = 13
    d["dd.a.b"] = 14
    d["dd.b.a"] = 15
    d["dd.b.b"] = 16

    assert d.build() == {
        "ll": [[1, 2], [3, 4]],
        "ld": [{"a": 5, "b": 6}, {"a": 7, "b": 8}],
        "dl": {"a": [9, 10], "b": [11, 12]},
        "dd": {"a": {"a": 13, "b": 14}, "b": {"a": 15, "b": 16}},
    }


def test_smart_split_dict_nested_list():
    d = SmartSplitDict()
    d["r.k1.0"] = "lol"
    d["r.k1.1"] = "lol"
    d["r.k2.0"] = "lol"
    d["r.k2.1"] = "lol"

    assert d.build() == {"r": {"k1": ["lol", "lol"], "k2": ["lol", "lol"]}}


def test_smart_split_dict_with_type():
    d = SmartSplitDict(sep=".")
    d["server"] = "fastapi"
    d["server.port"] = 8080
    assert d.build() == {"server": {"type": "fastapi", "port": 8080}}


def test_smart_split_dict_prebuilt():
    d = SmartSplitDict(sep=".")
    d["a.b.c"] = 1
    d["a"] = {"b": {"d": 2}}
    assert d.build() == {"a": {"b": {"c": 1, "d": 2}}}


def test_smart_split_dict_list_with_type():
    d = SmartSplitDict(sep=".")
    d["server.0"] = "fastapi"
    d["server.0.port"] = 8080
    assert d.build() == {"server": [{"type": "fastapi", "port": 8080}]}


def test_smart_split_dict_dict_with_type():
    d = SmartSplitDict(sep=".")
    d["server.a"] = "fastapi"
    d["server.a.port"] = 8080
    d["server.b"] = "fastapi"
    d["server.b.port"] = 8080
    assert d.build() == {
        "server": {
            "a": {"type": "fastapi", "port": 8080},
            "b": {"type": "fastapi", "port": 8080},
        }
    }
