"""美肌処理: シミ・ニキビ除去、肌のなめらか化 (質感を残す周波数分離)、肌色ムラ補正。"""

from __future__ import annotations

import cv2
import numpy as np

from ..settings import SkinSettings
from .ops import bgr_to_lab, blend, blur, guided_filter, lab_to_bgr, luminance, mask_roi, masked_blur


def remove_blemishes(img: np.ndarray, skin: np.ndarray, scale: float, amount: float) -> np.ndarray:
    """周囲より暗い / 赤い小さな斑点を検出し、周辺の肌で埋める。"""
    if amount <= 0:
        return img
    a = amount / 100.0
    pad = int(scale * 0.1) + 4
    roi = mask_roi(skin, pad, 0.3)
    if roi is None:
        return img
    sub, m = img[roi], skin[roi]

    lab = bgr_to_lab(sub)
    L, A = lab[..., 0], lab[..., 1]
    spot_sigma = max(1.5, scale * 0.02)
    bg_L = masked_blur(L, m, spot_sigma)
    bg_A = masked_blur(A, m, spot_sigma)
    dark = bg_L - L  # 周囲より暗い
    red = A - bg_A  # 周囲より赤い
    thr_dark = 7.0 - 3.5 * a
    thr_red = 6.0 - 2.5 * a
    cand = (((dark > thr_dark) | ((red > thr_red) & (dark > 1.0))) & (m > 0.5)).astype(np.uint8)

    # 小さく丸いものだけを斑点とみなす (眉毛・髪・しわ・輪郭は除外)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    max_area = (scale * (0.012 + 0.018 * a)) ** 2 * np.pi
    min_area = max(2, int((scale * 0.002) ** 2))
    keep = np.zeros(n, bool)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if not (min_area <= area <= max_area):
            continue
        aspect = max(w, h) / max(1, min(w, h))
        fill = area / max(1, w * h)
        keep[i] = aspect < 3.0 and fill > 0.3
    spots = keep[labels].astype(np.uint8)
    if not spots.any():
        return img

    grow = max(1, int(scale * 0.006))
    spots = cv2.dilate(spots, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1)))
    spot_f = spots.astype(np.float32)
    # 斑点を除いた周辺画素から塗りつぶし、細かい肌理 (テクスチャ) を少し戻す
    fill = masked_blur(sub, (1.0 - spot_f) * np.maximum(m, 0.05), max(2.0, scale * 0.015))
    tex = sub - cv2.GaussianBlur(sub, (0, 0), max(0.8, scale * 0.002))
    fill = fill + tex * 0.3
    alpha = cv2.GaussianBlur(spot_f, (0, 0), max(0.8, grow * 0.8)) * min(1.0, 0.6 + a)

    out = img.copy()
    out[roi] = blend(sub, fill, alpha)
    return out


def smooth_skin(img: np.ndarray, skin: np.ndarray, scale: float, smooth: float, texture: float) -> np.ndarray:
    """周波数分離による美肌。

    - 低〜中周波 (凹凸・毛穴の影・色ムラ) は Guided Filter でなめらかにする
    - 高周波 (肌理・産毛) は texture の割合だけ元画像から戻して、のっぺり感を防ぐ
    """
    if smooth <= 0:
        return img
    s = smooth / 100.0
    pad = int(scale * 0.08) + 4
    roi = mask_roi(skin, pad)
    if roi is None:
        return img
    sub, m = img[roi], skin[roi]

    # 大きい顔は縮小して処理 (半径がスケールに比例するため品質はほぼ同じ)
    work_scale = min(1.0, 600.0 / max(scale, 1.0))
    h, w = sub.shape[:2]
    small = cv2.resize(sub, (max(1, int(w * work_scale)), max(1, int(h * work_scale))), interpolation=cv2.INTER_AREA) if work_scale < 1 else sub
    radius = max(2, int(scale * work_scale * (0.012 + 0.018 * s)))
    eps = (0.012 + 0.03 * s) ** 2
    smoothed = guided_filter(luminance(small), small, radius, eps)
    if work_scale < 1:
        smoothed = cv2.resize(smoothed, (w, h), interpolation=cv2.INTER_LINEAR)

    fine_sigma = max(0.6, scale * 0.0025)
    fine = sub - cv2.GaussianBlur(sub, (0, 0), fine_sigma)
    result = smoothed + fine * (texture / 100.0)

    out = img.copy()
    out[roi] = blend(sub, result, m * s)
    return out


def even_skin_tone(img: np.ndarray, skin: np.ndarray, scale: float, amount: float) -> np.ndarray:
    """赤み・くすみなど中〜大きなスケールの色ムラを平均化する (肌理は保持)。"""
    if amount <= 0:
        return img
    a = amount / 100.0
    roi = mask_roi(skin, int(scale * 0.1) + 4)
    if roi is None:
        return img
    sub, m = img[roi], skin[roi]
    lab = bgr_to_lab(sub)
    sig = max(3.0, scale * 0.08)
    mb = np.clip(m, 0, 1)
    target = masked_blur(lab, mb, sig)
    mid = masked_blur(lab, mb, max(1.0, scale * 0.012))
    corr = (target - mid)
    corr[..., 0] *= 0.35  # 明るさの陰影 (立体感) はあまり消さない
    lab = lab + corr * (a * 0.9)
    out = img.copy()
    out[roi] = blend(sub, lab_to_bgr(lab), m)
    return out


def brighten_skin(img: np.ndarray, skin: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0:
        return img
    k = amount / 100.0 * 0.3
    lifted = img + (1.0 - img) * k * (0.5 + 0.5 * img)
    return blend(img, lifted, skin)


def apply(img: np.ndarray, skin: np.ndarray | None, scale: float, s: SkinSettings) -> np.ndarray:
    if skin is None:
        return img
    img = remove_blemishes(img, skin, scale, s.blemish)
    img = smooth_skin(img, skin, scale, s.smooth, s.texture)
    img = even_skin_tone(img, skin, scale, s.even_tone)
    img = brighten_skin(img, skin, s.brighten)
    return img
