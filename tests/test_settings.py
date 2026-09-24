import pytest

from picme.cli import _apply_overrides
from picme.settings import PRESETS, RetouchSettings, get_preset


def test_roundtrip(tmp_path):
    s = PRESETS["portrait"].copy()
    s.skin.smooth = 42
    path = tmp_path / "p.json"
    s.save(path)
    loaded = RetouchSettings.load(path)
    assert loaded.to_dict() == s.to_dict()


def test_from_dict_ignores_unknown_and_fills_defaults():
    s = RetouchSettings.from_dict({"skin": {"smooth": 10, "unknown": 5}, "extra": {}})
    assert s.skin.smooth == 10
    assert s.face.slim == 0
    assert s.color.auto_white_balance is True


def test_get_preset_returns_copy():
    s = get_preset("natural")
    s.skin.smooth = 99
    assert PRESETS["natural"].skin.smooth != 99


def test_get_preset_unknown():
    with pytest.raises(KeyError):
        get_preset("no-such-preset")


def test_cli_overrides():
    s = _apply_overrides(get_preset("none"), ["skin.smooth=60", "color.auto_tone=true"])
    assert s.skin.smooth == 60
    assert s.color.auto_tone is True
    with pytest.raises(SystemExit):
        _apply_overrides(s, ["skin.nope=1"])
