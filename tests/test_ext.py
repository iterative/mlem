from mlem.ext import MLEM_ENTRY_POINT, find_implementations, load_entrypoints


def test_load_entrypoints():
    exts = load_entrypoints()
    assert len(exts) > 0
    for e in exts.values():
        assert e.ep.module_name.startswith(MLEM_ENTRY_POINT)


def test_find_implementations():
    impls = find_implementations()
    assert MLEM_ENTRY_POINT in impls
    impls = impls[MLEM_ENTRY_POINT]
    for i in impls:
        assert not i.startswith("None")


def test_all_impls_in_entrypoints():
    # if this test fails, add new entrypoints (take the result of find_implementations()) to setup.py and
    # reinstall your dev copy of mlem to re-populate them
    exts = load_entrypoints()
    exts = {e.entry for e in exts.values()}
    impls = set(find_implementations()[MLEM_ENTRY_POINT])
    assert exts == impls
