"""基本補正: 自動ホワイトバランス・自動トーン・露出/コントラスト/色温度など。"""

from __future__ import annotations

import cv2
import numpy as np

from ..settings import ColorSettings
from .ops import blur, clip01, luminance, smoothstep, to_float, to_uint8


def _stats_image(img: np.ndarray, max_side: int = 512) -> np.ndarray:
    h, w = img.shape[:2]
    k = min(1.0, max_side / max(h, w))
    return cv2.resize(img, (max(1, int(w * k)), max(1, int(h * k))), interpolation=cv2.INTER_AREA) if k < 1 else img


def auto_white_balance(
    img: np.ndarray, strength: float = 0.8, skin_mask: np.ndarray | None = None, stats_img: np.ndarray | None = None
) -> np.ndarray:
    """無彩色に近い画素 (白い壁・シャツ・グレーなど) から色かぶりを推定して補正する。

    画面全体の平均を使う Gray-World 法は、暖色の背景や肌が多い写真で
    「本来の色」まで打ち消して青白くしてしまうため、元々ほぼ無彩色の
    画素だけを統計に使う。そうした画素が少ない写真ではほとんど補正しない。
    """
    small = _stats_image(img) if stats_img is None else stats_img
    px = small.reshape(-1, 3)
    lum = luminance(small).reshape(-1)
    chroma = (px.max(axis=1) - px.min(axis=1)) / np.maximum(lum, 1e-3)
    valid = (lum > 0.15) & (lum < 0.95) & (chroma < 0.35)
    if skin_mask is not None:
        sm = cv2.resize(skin_mask, (small.shape[1], small.shape[0])).reshape(-1)
        valid &= sm < 0.3
    frac = valid.mean()
    if frac < 0.01:
        return img
    # 無彩色に近いほど重く (かぶりの推定に信頼できる) 、明るい画素ほど重く
    w = (1.0 - chroma[valid] / 0.35) * lum[valid]
    mean = (px[valid] * w[:, None]).sum(axis=0) / max(w.sum(), 1e-6)
    gains = mean.mean() / np.maximum(mean, 1e-4)
    gains = np.clip(gains, 0.85, 1.2)
    confidence = min(1.0, frac / 0.05)  # 無彩色の画素が少ないときは控えめに
    gains = 1.0 + (gains - 1.0) * strength * confidence
    gains /= gains @ np.array([0.114, 0.587, 0.299])  # 明るさは変えない
    return img * gains.astype(np.float32)


def sharpen(img: np.ndarray, amount: float, protect: np.ndarray | None = None) -> np.ndarray:
    """輝度のアンシャープマスク。protect (0..1) の領域は弱める (肌など)。"""
    if amount <= 0:
        return img
    sigma = max(0.8, min(img.shape[:2]) / 1200.0)
    lum = luminance(img)
    detail = cv2.subtract(lum, cv2.GaussianBlur(lum, (0, 0), sigma)) * np.float32(amount / 100.0 * 1.2)
    if protect is not None:
        detail = cv2.multiply(detail, 1.0 - np.float32(0.8) * protect.astype(np.float32, copy=False))
    return cv2.add(img, cv2.merge([detail] * 3))


def _tone_params(small: np.ndarray, strength: float = 0.8) -> tuple[float, float, float] | None:
    """auto_tone と同じ黒点・白点・ガンマを統計用の小さな画像から求める。"""
    lum = luminance(small)
    lo, hi = np.percentile(lum, [0.5, 99.7])
    lo = min(lo, 0.12) * strength
    hi = 1.0 - (1.0 - max(hi, 0.8)) * strength
    if hi - lo < 0.2:
        return None
    mid = float(np.median(np.clip((lum - lo) / (hi - lo), 1e-3, 1)))
    gamma = np.clip(np.log(0.46) / np.log(max(mid, 1e-3)), 0.7, 1.3)
    return float(lo), float(hi), float(1.0 + (gamma - 1.0) * strength * 0.7)


def build_curves(img_u8: np.ndarray, s: ColorSettings, skin_mask: np.ndarray | None = None) -> np.ndarray:
    """チャンネルごとのトーンカーブ (WB → 自動トーン → 露出 → コントラスト → 色温度) を
    256 段の LUT にまとめる。画素ごとの演算が 1 回で済むので大きな画像でも速い。"""
    x = np.tile(np.arange(256, dtype=np.float32)[:, None] / 255.0, (1, 3))  # (256, 3) BGR
    small = to_float(_stats_image(img_u8))
    if s.auto_white_balance:
        ones = np.ones((1, 1, 3), np.float32)
        gains = auto_white_balance(ones, skin_mask=skin_mask, stats_img=small).reshape(3)
        x = x * gains
        small = small * gains
    if s.auto_tone and (params := _tone_params(small)) is not None:
        lo, hi, gamma = params
        x = np.power(np.clip((x - lo) / (hi - lo), 0, None), gamma)
    if s.exposure:
        x = x * np.float32(2.0 ** (s.exposure / 50.0))
    if s.contrast:
        c = s.contrast / 100.0
        x = np.clip(x, 0, 1)
        if c > 0:
            x = x + (x * x * (3.0 - 2.0 * x) - x) * c  # S カーブ
        else:
            x = x + ((0.5 + (x - 0.5) * 0.5) - x) * (-c)
    if s.temperature or s.tint:
        t, m = s.temperature / 100.0, s.tint / 100.0
        g = np.array([1.0 - 0.15 * t, 1.0 - 0.10 * m, 1.0 + 0.15 * t], np.float32)
        x = x * (g / (g @ np.array([0.114, 0.587, 0.299], np.float32)))
    return np.ascontiguousarray(x.astype(np.float32).reshape(1, 256, 3))


def local_tone(img: np.ndarray, s: ColorSettings) -> np.ndarray:
    """ハイライト / シャドウ (局所的な明るさを基準にした補正)。"""
    if not (s.highlights or s.shadows):
        return img
    lum = clip01(luminance(img))
    # 局所的な明るさを基準にするとハロが出にくい自然なトーン補正になる
    base = blur(lum, max(img.shape[:2]) * 0.01)
    w_sh = (1.0 - smoothstep(0.0, 0.55, base)) ** 2
    w_hl = smoothstep(0.45, 1.0, base) ** 2
    delta = np.float32(0.35 * s.shadows / 100.0) * w_sh + np.float32(0.35 * s.highlights / 100.0) * w_hl
    new = clip01(lum + delta * np.where(delta > 0, 1.0 - lum, lum))
    ratio = np.minimum(new / np.maximum(lum, np.float32(1e-3)), np.float32(4.0))
    img *= ratio[..., None]
    return img


def saturation(img: np.ndarray, s: ColorSettings) -> np.ndarray:
    if not (s.vibrance or s.saturation):
        return img
    gray = luminance(img)[..., None]
    factor: np.ndarray | np.float32 = np.float32(1.0 + s.saturation / 100.0)
    if s.vibrance:
        b, g, r = cv2.split(img)
        sat = cv2.subtract(cv2.max(cv2.max(b, g), r), cv2.min(cv2.min(b, g), r))
        factor = factor + np.float32(s.vibrance / 100.0) * clip01(1.0 - sat * np.float32(1.5))
        factor = factor[..., None]
    img -= gray  # その場で計算してメモリの読み書きを減らす
    img *= factor
    img += gray
    return img


def apply(img: np.ndarray, s: ColorSettings, skin_mask: np.ndarray | None = None) -> np.ndarray:
    """基本補正。img は BGR uint8 (推奨) または float32 (0..1)。float32 (0..1) を返す。"""
    img_u8 = img if img.dtype == np.uint8 else to_uint8(img)
    out = cv2.LUT(img_u8, build_curves(img_u8, s, skin_mask))
    out = local_tone(out, s)
    out = saturation(out, s)
    return clip01(out)
