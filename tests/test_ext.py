import re
from pathlib import Path

from mlem import ExtensionLoader
from mlem.config import MlemConfig, MlemConfigBase
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


def _write_entrypoints(impls_sorted):
    setup_path = Path(__file__).parent.parent / "setup.py"
    with open(setup_path, encoding="utf8") as f:
        setup_py = f.read()
    impls_string = ",\n".join(f'            "{i}"' for i in impls_sorted)
    new_entrypoints = f'"mlem.contrib": [\n{impls_string},\n        ]'
    setup_py = re.subn(
        r'"mlem\.contrib": \[\n[^]]*]', new_entrypoints, setup_py
    )[0]
    with open(setup_path, "w", encoding="utf8") as f:
        f.write(setup_py)


def test_all_impls_in_entrypoints():
    # if this test fails, add new entrypoints (take the result of find_implementations()) to setup.py and
    # reinstall your dev copy of mlem to re-populate them
    exts = load_entrypoints()
    exts = {e.entry for e in exts.values()}
    impls = find_abc_implementations()[MLEM_ENTRY_POINT]
    impls_sorted = sorted(
        impls, key=lambda x: tuple(x.split(" = ")[1].split(":"))
    )
    impls_set = set(impls)
    if exts != impls_set:
        _write_entrypoints(impls_sorted)
        assert (
            exts == impls_set
        ), "New enrtypoints written to setup.py, please reinstall"


def test_all_configs_in_entrypoints():
    impls = find_implementations(MlemConfigBase)
    impls[MlemConfig] = f"{MlemConfig.__module__}:{MlemConfig.__name__}"
    assert {
        e.entry for e in load_entrypoints(MLEM_CONFIG_ENTRY_POINT).values()
    } == {f"{i.__config__.section} = {k}" for i, k in impls.items()}


def test_all_ext_has_pip_extra():
    from setup import extras

    exts_reqs = {
        v.extra: v.reqs_packages
        for v in ExtensionLoader.builtin_extensions.values()
        if v.extra is not None and len(v.reqs_packages)
    }

    for name, reqs in exts_reqs.items():
        assert name in extras
        assert set(reqs) == set(extras[name])
