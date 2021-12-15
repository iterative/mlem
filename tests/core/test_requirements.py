import pytest
from pydantic import parse_obj_as

from mlem.contrib.sklearn import SklearnModel
from mlem.core.objects import ModelMeta
from mlem.core.requirements import (
    CustomRequirement,
    InstallableRequirement,
    Requirements,
    resolve_requirements,
)


def test_resolve_requirements_arg():
    requirements = Requirements.new(
        [
            InstallableRequirement(module="dumb", version="0.4.1"),
            InstallableRequirement(module="art", version="4.0"),
        ]
    )
    actual_reqs = resolve_requirements(requirements)
    assert actual_reqs == requirements


def test_resolve_requirement_arg():
    req = InstallableRequirement(module="dumb", version="0.4.1")
    actual_reqs = resolve_requirements(req)
    assert actual_reqs.installable[0] == req


def test_resolve_requirement_list_arg():
    req = [
        InstallableRequirement(module="dumb", version="0.4.1"),
        InstallableRequirement(module="art", version="4.0"),
    ]
    actual_reqs = resolve_requirements(req)
    assert len(actual_reqs.installable) == 2
    assert actual_reqs.installable == req


def test_resolve_str_arg():
    req = "dumb==0.4.1"
    actual_reqs = resolve_requirements(req)
    assert actual_reqs.installable[0].to_str() == req


def test_resolve_str_list_arg():
    req = ["dumb==0.4.1", "art==4.0"]
    actual_reqs = resolve_requirements(req)
    assert len(actual_reqs.installable) == 2
    assert sorted(req) == sorted([r.to_str() for r in actual_reqs.installable])


def test_installable_requirement__from_module():
    import pandas as pd

    assert (
        InstallableRequirement.from_module(pd).to_str()
        == f"pandas=={pd.__version__}"
    )

    import numpy as np

    assert (
        InstallableRequirement.from_module(np).to_str()
        == f"numpy=={np.__version__}"
    )

    import sklearn as sk

    assert (
        InstallableRequirement.from_module(sk).to_str()
        == f"scikit-learn=={sk.__version__}"
    )
    assert (
        InstallableRequirement.from_module(sk, "xyz").to_str()
        == f"xyz=={sk.__version__}"
    )


def test_custom_requirement__source():
    from mlem import core

    package = CustomRequirement.from_module(core)
    assert package.is_package
    assert package.sources is not None
    with pytest.raises(AttributeError):
        package.source  # pylint: disable=pointless-statement

    module = CustomRequirement.from_module(core.requirements)
    assert not module.is_package
    assert module.source is not None
    with pytest.raises(AttributeError):
        module.sources  # pylint: disable=pointless-statement


def test_resolve_unique_str():
    reqs_str = ["a==1", "a==1"]
    reqs = Requirements.new(reqs_str)
    assert len(reqs.__root__) == 1
    assert reqs.installable[0] == InstallableRequirement(
        module="a", version="1"
    )


def test_resolve_unique_req():
    req = InstallableRequirement(module="a", version="1")
    reqs_list = [req, req]
    reqs = Requirements.new(reqs_list)
    assert len(reqs.__root__) == 1
    assert reqs.installable[0] == req


def test_serialize_empty():
    mt = SklearnModel(methods={}, model="")
    obj = ModelMeta(model_type=mt)
    payload = obj.dict()
    obj2 = ModelMeta(model_type=mt)
    obj2.requirements.__root__.append(InstallableRequirement(module="sklearn"))
    assert obj.requirements.__root__ == []
    new_obj = parse_obj_as(ModelMeta, payload)
    assert new_obj == obj


# Copyright 2019 Zyfra
# Copyright 2021 Iterative
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
