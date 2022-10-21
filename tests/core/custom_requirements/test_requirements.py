import os
import shutil
import subprocess

import dill
from pack_1 import TestM

from mlem.core.metadata import get_object_metadata
from mlem.core.objects import MlemModel
from mlem.utils.module import get_object_requirements


def test_requirements_analyzer__custom_modules():
    """Import chains:
    proxy_model (skipped) -> {pandas(skipped), model_trainer}
    model_trainer -> {six, sklearn, proxy_pkg_import (local import in function)}
    proxy_pkg_import -> pkg_import
    pkg_import -> pkg
    pkg -> all of pkg
    pkg.subpkg.impl -> isort
    """
    import catboost  # pylint: disable=unused-import # noqa
    import unused_code  # pylint: disable=unused-import # noqa
    from proxy_model import model

    reqs = get_object_requirements(model)

    custom_reqs = {req.name for req in reqs.custom}
    # "test_cases" appears here as this code is imported by pytest
    # __main__ modules won't appear here
    assert {
        "model_trainer",
        "proxy_pkg_import",
        "pkg_import",
        "pkg",
    } == custom_reqs

    inst_reqs = {req.package for req in reqs.installable}
    assert {"scikit-learn", "six", "isort"} == inst_reqs


def test_requirements_analyzer__model_works(tmpdir):
    from proxy_model import model

    reqs = get_object_requirements(model)

    reqs.materialize_custom(tmpdir)
    assert os.path.exists(
        os.path.join(tmpdir, "pkg", "subpkg", "testfile.json")
    )

    with open(os.path.join(tmpdir, "model.pkl"), "wb") as f:
        dill.dump(model, f)

    shutil.copy(
        os.path.join(os.path.dirname(__file__), "use_model.py"), tmpdir
    )

    cp = subprocess.run(
        "python use_model.py", shell=True, cwd=tmpdir, check=False
    )
    assert cp.returncode == 0


def test_model_custom_requirements(tmpdir):
    from pack_1.model_type import (  # pylint: disable=unused-import  # noqa
        TestModelType,
    )

    model = get_object_metadata(TestM(), 1)
    assert isinstance(model, MlemModel)

    model.dump(os.path.join(tmpdir, "model"))
    model.requirements.materialize_custom(tmpdir)
    shutil.copy(
        os.path.join(os.path.dirname(__file__), "use_model_meta.py"), tmpdir
    )

    cp = subprocess.run(
        "python use_model_meta.py", shell=True, cwd=tmpdir, check=False
    )
    assert cp.returncode == 0, cp.stderr

    assert {x.name for x in model.requirements.custom} == {"pack_1", "pack_2"}
    assert {x.module for x in model.requirements.installable} == {"numpy"}


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
