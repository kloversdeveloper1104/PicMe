import numpy as np

from picme.analysis import Analysis, Face
from picme.pipeline import face_plan, match_override, retouch
from picme.retouch import background, hair, heal
from picme.retouch.ops import at_scale, blend, mask_roi, polygon_mask_in, polygons_roi, to_float
from picme.settings import BackgroundSettings, FaceOverride, FaceSettings, RetouchSettings, get_preset


def _fake_face(cx: float, cy: float, size: float) -> Face:
    """中心 (cx, cy)、幅 size の円周上に 478 点を並べたダミーの顔。"""
    t = np.linspace(0, 2 * np.pi, 478, endpoint=False)
    pts = np.stack([cx + np.cos(t) * size / 2, cy + np.sin(t) * size / 2], axis=1).astype(np.float32)
    return Face(pts)


def test_settings_roundtrip_with_image_data(tmp_path):
    s = get_preset("portrait")
    s.background.replace_color = "#ffffff"
    s.face_overrides = [FaceOverride(0.3, 0.4, FaceSettings(slim=50), skin=40)]
    s.heal_spots = [[0.5, 0.5, 0.01]]
    d = s.to_dict()
    back = RetouchSettings.from_dict(d)
    assert back.to_dict() == d
    # プリセットとして保存するときは画像固有のデータを含めない
    p = tmp_path / "p.json"
    s.save(p)
    loaded = RetouchSettings.load(p)
    assert loaded.face_overrides == [] and loaded.heal_spots == []
    assert loaded.background.replace_color == "#ffffff"
    assert s.preset_only().heal_spots == []


def test_match_override_by_position():
    f = _fake_face(100, 100, 80)
    near = FaceOverride(100 / 400, 104 / 300)
    far = FaceOverride(0.9, 0.9)
    assert match_override(f, [far, near], (300, 400)) is near
    assert match_override(f, [far], (300, 400)) is None


def test_face_plan_uses_override_and_skin_gain():
    a = Analysis(shape=(300, 400), faces=[_fake_face(100, 150, 100), _fake_face(300, 150, 100)])
    s = get_preset("none")
    s.face.slim = 30
    s.face_overrides = [FaceOverride(100 / 400, 150 / 300, FaceSettings(slim=0), skin=0)]
    plan, gain = face_plan(a, s)
    assert plan[0][1].slim == 0 and plan[1][1].slim == 30
    assert gain[150, 100] < 0.05 and gain[150, 300] > 0.99


def test_heal_spot_removes_dark_dot():
    img = np.full((120, 120, 3), 0.7, np.float32)
    img += np.random.default_rng(0).normal(0, 0.01, img.shape).astype(np.float32)
    img[55:65, 55:65] = 0.2
    out = heal.apply(img, [[0.5, 0.5, 0.06]])
    assert out[58:62, 58:62].mean() > 0.6
    assert np.allclose(out[:20], img[:20])  # 離れた場所は変わらない


def test_background_replace_color():
    img = np.full((50, 50, 3), 0.2, np.float32)
    person = np.zeros((50, 50), np.float32)
    person[10:40, 10:40] = 1.0
    out = background.apply(img, person, BackgroundSettings(replace_color="#ff0000"))
    assert np.allclose(out[2, 2], [0, 0, 1], atol=1e-3)  # BGR で赤
    assert np.allclose(out[25, 25], 0.2, atol=1e-3)
    assert background.parse_color("255,128,0") is not None
    assert background.parse_color("zz") is None


def test_background_replace_image(tmp_path):
    from PIL import Image

    p = tmp_path / "bg.png"
    Image.fromarray(np.full((10, 20, 3), (0, 255, 0), np.uint8)).save(p)
    img = np.full((40, 40, 3), 0.5, np.float32)
    person = np.zeros((40, 40), np.float32)
    out = background.apply(img, person, BackgroundSettings(replace_image=str(p)))
    assert np.allclose(out[5, 5], [0, 1, 0], atol=1e-3)


def test_hair_shine_only_inside_hair():
    rng = np.random.default_rng(0)
    img = rng.uniform(0.1, 0.4, (80, 80, 3)).astype(np.float32)
    hair_mask = np.zeros((80, 80), np.float32)
    hair_mask[:40] = 1.0
    out = hair.add_shine(img, hair_mask, 80, 100)
    assert not np.allclose(out[:30], img[:30])
    assert np.allclose(out[60:], img[60:])


def test_retouch_with_all_features_on_fake_faces():
    rng = np.random.default_rng(0)
    img = (rng.uniform(0.3, 0.7, (240, 320, 3)) * 255).astype(np.uint8)
    a = Analysis(
        shape=(240, 320),
        faces=[_fake_face(100, 120, 90), _fake_face(230, 120, 90)],
        skin_mask=np.ones((240, 320), np.float32) * 0.8,
        person_mask=np.ones((240, 320), np.float32) * 0.9,
        hair_mask=np.zeros((240, 320), np.float32),
    )
    s = get_preset("wedding")
    s.background.replace_color = "#ffffff"
    s.face_overrides = [FaceOverride(100 / 320, 0.5, FaceSettings(eye_enlarge=50), skin=50)]
    s.heal_spots = [[0.5, 0.5, 0.02]]
    out = retouch(img, a, s)
    assert out.shape == img.shape and out.dtype == np.uint8


def test_ops_helpers():
    img = to_float(np.random.default_rng(0).integers(0, 255, (60, 80, 3), dtype=np.uint8))
    assert np.allclose(blend(img, img * 0, 0.0), img)
    assert np.allclose(blend(img, img * 0, np.ones((60, 80), np.float32)), 0)
    m = np.zeros((60, 80), np.float32)
    m[10:20, 30:40] = 1
    assert mask_roi(m, 0) == (slice(10, 20), slice(30, 40))
    assert mask_roi(np.zeros((5, 5), np.float32), 1) is None
    poly = np.array([[30, 10], [40, 10], [40, 20], [30, 20]], np.float32)
    roi = polygons_roi((60, 80), [poly], 2)
    assert polygon_mask_in(roi, [poly]).max() == 1.0
    # at_scale: 定数を足す処理は縮小しても同じ結果
    out = at_scale(lambda x: x + 0.1, img, 0.5)
    assert np.allclose(out, img + 0.1, atol=1e-5)
