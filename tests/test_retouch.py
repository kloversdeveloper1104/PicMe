import cv2
import numpy as np

from picme.analysis import Analysis, FaceAnalyzer, color_skin_mask
from picme.pipeline import retouch
from picme.retouch import color, skin
from picme.retouch.face import WarpField
from picme.retouch.ops import guided_filter, to_float
from picme.settings import PRESETS, get_preset


def test_none_preset_is_identity(portrait_like):
    a = FaceAnalyzer(use_ai=False).analyze(portrait_like)
    out = retouch(portrait_like, a, get_preset("none"))
    assert out.shape == portrait_like.shape and out.dtype == np.uint8
    assert np.abs(out.astype(int) - portrait_like.astype(int)).max() <= 1


def test_all_presets_run_without_ai(portrait_like):
    a = FaceAnalyzer(use_ai=False).analyze(portrait_like)
    assert not a.faces and a.skin_mask is not None
    for name, preset in PRESETS.items():
        out = retouch(portrait_like, a, preset)
        assert out.shape == portrait_like.shape, name


def test_analysis_resize_between_resolutions(portrait_like):
    a = FaceAnalyzer(use_ai=False).analyze(portrait_like)
    big = cv2.resize(portrait_like, None, fx=2, fy=2)
    out = retouch(big, a, get_preset("natural"))  # 低解像度の解析結果を流用できる
    assert out.shape == big.shape


def test_skin_smoothing_reduces_noise(portrait_like):
    img = to_float(portrait_like)
    mask = color_skin_mask(portrait_like)
    out = skin.smooth_skin(img, mask, scale=180, smooth=100, texture=0)
    inner = (slice(140, 220), slice(110, 190))
    assert out[inner].std(axis=(0, 1)).mean() < img[inner].std(axis=(0, 1)).mean() * 0.6


def test_blemish_removal_fills_dark_spot():
    img = np.full((200, 200, 3), (140, 170, 220), np.uint8)
    cv2.circle(img, (100, 100), 4, (70, 80, 110), -1)
    mask = np.ones((200, 200), np.float32)
    out = skin.remove_blemishes(to_float(img), mask, scale=200, amount=100)
    before = to_float(img)[100, 100].mean()
    after = out[100, 100].mean()
    ref = to_float(img)[20, 20].mean()
    assert abs(after - ref) < abs(before - ref) * 0.3


def test_auto_white_balance_neutralizes_cast():
    rng = np.random.default_rng(1)
    gray = rng.uniform(0.2, 0.8, (100, 100, 1)).astype(np.float32)
    cast = gray * np.array([0.85, 1.0, 1.15], np.float32)  # 赤かぶり
    out = color.auto_white_balance(cast, strength=1.0)
    spread_before = np.ptp(cast.reshape(-1, 3).mean(0))
    spread_after = np.ptp(out.reshape(-1, 3).mean(0))
    assert spread_after < spread_before * 0.5


def test_exposure_brightens():
    img = np.full((10, 10, 3), 0.3, np.float32)
    s = get_preset("none").color
    s.exposure = 50
    assert color.adjust_tone(img, s).mean() > 0.55


def test_guided_filter_constant_image():
    img = np.full((20, 20, 3), 0.5, np.float32)
    out = guided_filter(img[..., 0], img, 3, 1e-3)
    assert np.allclose(out, 0.5, atol=1e-4)


def test_warp_field_identity_and_push():
    img = np.zeros((100, 100, 3), np.float32)
    img[:, 50:] = 1.0
    field = WarpField((100, 100))
    assert field.apply(img) is img
    field.push(np.array([50, 50], np.float32), np.array([10, 0], np.float32), 30)
    out = field.apply(img)
    # 境界が右へ押し出されるので、中心の少し右は黒くなる
    assert out[50, 55].mean() < 0.5
    # 影響半径の外は変化しない
    assert np.array_equal(out[5], img[5])


def test_background_blur_requires_person_mask(portrait_like):
    a = Analysis(shape=portrait_like.shape[:2])
    s = get_preset("none")
    s.background.blur = 100
    out = retouch(portrait_like, a, s)
    assert np.abs(out.astype(int) - portrait_like.astype(int)).max() <= 1
    a.person_mask = np.zeros(portrait_like.shape[:2], np.float32)
    out = retouch(portrait_like, a, s)
    assert out.astype(np.float32).std() < portrait_like.astype(np.float32).std()
