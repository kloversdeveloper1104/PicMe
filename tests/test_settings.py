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
    s = _apply_overrides(get_preset("none"), ["skin.smooth=60", "color.auto_tone=true", "background.replace_color=#ffffff"])
    assert s.skin.smooth == 60
    assert s.background.replace_color == "#ffffff"
    assert s.color.auto_tone is True
    with pytest.raises(SystemExit):
        _apply_overrides(s, ["skin.nope=1"])


def test_history_undo_redo():
    from picme.gui.history import History

    h = History(limit=3)
    h.reset({"v": 0})
    assert not h.commit({"v": 0})  # 変化なしは積まない
    for v in (1, 2, 3, 4):
        assert h.commit({"v": v})
    assert h.undo() == {"v": 3}
    assert h.undo() == {"v": 2}
    assert h.redo() == {"v": 3}
    h.commit({"v": 9})
    assert not h.can_redo
    assert h.undo() == {"v": 3} and h.undo() == {"v": 2} and h.undo() == {"v": 1}
    assert h.undo() is None  # limit=3 を超えた古い履歴は捨てられている
