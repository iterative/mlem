from mlem.telemetry import telemetry


def test_is_enabled():
    assert not telemetry.is_enabled()
