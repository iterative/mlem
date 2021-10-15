from mlem.analytics import is_enabled


def test_is_enabled():
    assert not is_enabled()
