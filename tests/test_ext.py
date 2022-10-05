import os
import re
from importlib import import_module
from pathlib import Path

import pytest

from mlem.config import MlemConfig, MlemConfigBase
from mlem.ext import ExtensionLoader, get_ext_type
from mlem.utils.entrypoints import (
    MLEM_CONFIG_ENTRY_POINT,
    MLEM_ENTRY_POINT,
    find_abc_implementations,
    find_implementations,
    load_entrypoints,
)


def test_load_entrypoints():
    exts = load_entrypoints()
    assert len(exts) > 0
    for e in exts.values():
        assert e.ep.module_name.startswith(MLEM_ENTRY_POINT)


def test_find_implementations():
    impls = find_abc_implementations()
    assert MLEM_ENTRY_POINT in impls
    impls = impls[MLEM_ENTRY_POINT]
    for i in impls:
        assert not i.startswith("None")


def _write_entrypoints(impls_sorted, section: str):
    setup_path = Path(__file__).parent.parent / "setup.py"
    with open(setup_path, encoding="utf8") as f:
        setup_py = f.read()
    impls_string = ",\n".join(f'            "{i}"' for i in impls_sorted)
    new_entrypoints = f'"{section}": [\n{impls_string},\n        ]'
    setup_py = re.subn(rf'"{section}": \[\n[^]]*]', new_entrypoints, setup_py)[
        0
    ]
    with open(setup_path, "w", encoding="utf8") as f:
        f.write(setup_py)


def test_all_impls_in_entrypoints():
    # if this test fails, add new entrypoints (take the result of find_implementations()) to setup.py and
    # reinstall your dev copy of mlem to re-populate them
    exts = load_entrypoints()
    exts = {e.entry for e in exts.values()}
    impls = find_abc_implementations(raise_on_error=True)[MLEM_ENTRY_POINT]
    impls_sorted = sorted(
        impls, key=lambda x: tuple(x.split(" = ")[1].split(":"))
    )
    impls_set = set(impls)
    if exts != impls_set:
        _write_entrypoints(impls_sorted, "mlem.contrib")
        assert (
            exts == impls_set
        ), "New enrtypoints written to setup.py, please reinstall"


def test_all_configs_in_entrypoints():
    impls = find_implementations(MlemConfigBase, raise_on_error=True)
    impls[MlemConfig] = f"{MlemConfig.__module__}:{MlemConfig.__name__}"
    impls_sorted = sorted(
        {f"{i.__config__.section} = {k}" for i, k in impls.items()},
        key=lambda x: tuple(x.split(" = ")[1].split(":")),
    )
    exts = {
        e.entry for e in load_entrypoints(MLEM_CONFIG_ENTRY_POINT).values()
    }
    if exts != set(impls_sorted):
        _write_entrypoints(impls_sorted, "mlem.config")
        assert exts == impls_sorted


def test_all_ext_has_pip_extra():
    from setup import extras

    exts_reqs = {
        v.extra: v.reqs_packages
        for v in ExtensionLoader.builtin_extensions.values()
        if v.extra is not None and len(v.reqs_packages)
    }

    for name, reqs in exts_reqs.items():
        assert name in extras
        ext_extras = extras[name]
        assert set(reqs) == {re.split("[~=]", r)[0] for r in ext_extras}


def test_all_ext_registered():
    from mlem import contrib

    files = os.listdir(os.path.dirname(contrib.__file__))
    ext_sources = {
        name[: -len(".py")] if name.endswith(".py") else name
        for name in files
        if not name.startswith("__")
    }
    assert set(ExtensionLoader.builtin_extensions) == {
        f"mlem.contrib.{name}" for name in ext_sources
    }


@pytest.mark.parametrize("mod", ExtensionLoader.builtin_extensions.keys())
def test_all_ext_docstring(mod):
    module = import_module(mod)
    assert module.__doc__ is not None
    assert get_ext_type(mod) in {
        "model",
        "deployment",
        "data",
        "serving",
        "build",
        "uri",
        "storage",
    }
