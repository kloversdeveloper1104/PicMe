"""美肌処理: シミ・ニキビ除去、肌のなめらか化 (質感を残す周波数分離)、肌色ムラ補正。"""

from __future__ import annotations

import cv2
import numpy as np

from .. import landmarks as lm
from ..analysis import Face
from ..settings import SkinSettings
from .ops import at_scale, bgr_to_lab, blend, blur, clip01, guided_filter, lab_to_bgr, luminance, mask_roi, masked_blur


WORK_FACE_SIZE = 500.0  # 低周波処理はこの顔サイズ (px) 相当まで縮小して計算する


def remove_blemishes(img: np.ndarray, skin: np.ndarray, scale: float, amount: float) -> np.ndarray:
    """周囲より暗い / 赤い小さな斑点を検出し、周辺の肌で埋める。"""
    if amount <= 0:
        return img
    pad = int(scale * 0.1) + 4
    roi = mask_roi(skin, pad, 0.3)
    if roi is None:
        return img
    k = min(1.0, WORK_FACE_SIZE / max(scale, 1.0))
    out = img.copy()
    out[roi] = at_scale(lambda sub, m: _remove_blemishes(sub, m, scale * k, amount), img[roi], k, skin[roi])
    return out


def _remove_blemishes(sub: np.ndarray, m: np.ndarray, scale: float, amount: float) -> np.ndarray:
    a = amount / 100.0

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
        return sub

    grow = max(1, int(scale * 0.006))
    spots = cv2.dilate(spots, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1)))
    spot_f = spots.astype(np.float32)
    # 斑点を除いた周辺画素から塗りつぶし、細かい肌理 (テクスチャ) を少し戻す
    fill = masked_blur(sub, (1.0 - spot_f) * np.maximum(m, 0.05), max(2.0, scale * 0.015))
    tex = sub - cv2.GaussianBlur(sub, (0, 0), max(0.8, scale * 0.002))
    fill = fill + tex * 0.3
    alpha = cv2.GaussianBlur(spot_f, (0, 0), max(0.8, grow * 0.8)) * min(1.0, 0.6 + a)
    return blend(sub, fill, alpha)


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
    work_scale = min(1.0, 400.0 / max(scale, 1.0))
    h, w = sub.shape[:2]
    small = cv2.resize(sub, (max(1, int(w * work_scale)), max(1, int(h * work_scale))), interpolation=cv2.INTER_AREA) if work_scale < 1 else sub
    radius = max(2, int(scale * work_scale * (0.012 + 0.018 * s)))
    eps = (0.012 + 0.03 * s) ** 2
    smoothed = guided_filter(luminance(small), small, radius, eps)
    if work_scale < 1:
        smoothed = cv2.resize(smoothed, (w, h), interpolation=cv2.INTER_LINEAR)

    fine_sigma = max(0.6, scale * 0.0025)
    fine = cv2.subtract(sub, cv2.GaussianBlur(sub, (0, 0), fine_sigma))
    result = cv2.scaleAdd(fine, texture / 100.0, smoothed.astype(np.float32, copy=False))

    out = img.copy()
    out[roi] = blend(sub, result, m * s)
    return out


def even_skin_tone(img: np.ndarray, skin: np.ndarray, scale: float, amount: float) -> np.ndarray:
    """赤み・くすみなど中〜大きなスケールの色ムラを平均化する (肌理は保持)。"""
    if amount <= 0:
        return img
    roi = mask_roi(skin, int(scale * 0.1) + 4)
    if roi is None:
        return img
    k = min(1.0, WORK_FACE_SIZE / max(scale, 1.0))
    out = img.copy()
    out[roi] = at_scale(lambda sub, m: _even_skin_tone(sub, m, scale * k, amount), img[roi], k, skin[roi])
    return out


def _even_skin_tone(sub: np.ndarray, m: np.ndarray, scale: float, amount: float) -> np.ndarray:
    a = amount / 100.0
    lab = bgr_to_lab(sub)
    mb = clip01(m)
    target = masked_blur(lab, mb, max(3.0, scale * 0.08))
    mid = masked_blur(lab, mb, max(1.0, scale * 0.012))
    corr = target - mid
    corr[..., 0] *= 0.35  # 明るさの陰影 (立体感) はあまり消さない
    lab = lab + corr * np.float32(a * 0.9)
    return blend(sub, lab_to_bgr(lab), m)


def brighten_skin(img: np.ndarray, skin: np.ndarray, amount: float) -> np.ndarray:
    """色相・彩度を保ったまま肌を明るくする (白を混ぜると暗い肌色がくすむため)。"""
    if amount <= 0:
        return img
    roi = mask_roi(skin, 2)
    if roi is None:
        return img
    k = amount / 100.0 * 0.35
    sub = img[roi]
    gain = 1.0 + k * (1.0 - clip01(luminance(sub))) ** 1.5
    out = img.copy()
    out[roi] = blend(sub, sub * gain[..., None], skin[roi])
    return out


def wrinkle_mask(shape: tuple[int, int], faces: list[Face]) -> np.ndarray:
    """しわが出やすい部位 (ほうれい線・額・目尻) のマスク。"""
    mask = np.zeros(shape, np.float32)
    for f in faces:
        sc = f.scale
        # ほうれい線: 小鼻の横 → 口角の外側 → 少し下
        axis = f.points[lm.CHIN] - f.points[lm.FOREHEAD]
        axis /= max(float(np.linalg.norm(axis)), 1e-3)
        for ala, corner in ((129, 61), (358, 291)):
            a, c = f.points[ala], f.points[corner]
            out = c - f.points[lm.NOSE_TIP]
            out = out - axis * float(out @ axis)
            out /= max(float(np.linalg.norm(out)), 1e-3)
            line = np.array([a + out * sc * 0.02, (a + c) / 2 + out * sc * 0.06, c + out * sc * 0.07 + axis * sc * 0.06])
            cv2.polylines(mask, [np.round(line).astype(np.int32)], False, 1.0, max(2, int(sc * 0.07)), cv2.LINE_AA)
        # 額: 眉の上から生え際まで (髪は肌マスクとの積で除外される)
        forehead = np.concatenate([f.pts([54, 103, 67, 109, 10, 338, 297, 332, 284]), f.pts([300, 293, 334, 296, 336, 107, 66, 105, 63, 70])])
        hull = cv2.convexHull(np.round(forehead).astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 1.0, cv2.LINE_AA)
        # 目尻 (カラスの足跡)
        for outer, inner in ((33, 133), (263, 362)):
            o, i = f.points[outer], f.points[inner]
            ew = float(np.linalg.norm(o - i))
            c = o + (o - i) / max(ew, 1e-3) * ew * 0.35
            cv2.circle(mask, (int(c[0]), int(c[1])), max(2, int(ew * 0.4)), 1.0, -1, cv2.LINE_AA)
    return mask


def reduce_wrinkles(img: np.ndarray, skin: np.ndarray, faces: list[Face], scale: float, amount: float) -> np.ndarray:
    """しわの溝 (周囲より暗い細い線) を持ち上げ、その部位だけを強めになめらかにする。"""
    if amount <= 0 or not faces:
        return img
    a = amount / 100.0
    region = wrinkle_mask(img.shape[:2], faces)
    region = cv2.GaussianBlur(region, (0, 0), max(1.0, scale * 0.02)) * np.clip(skin * 1.3, 0, 1)
    roi = mask_roi(region, int(scale * 0.05) + 4)
    if roi is None:
        return img
    sub, m = img[roi], region[roi]
    lum = luminance(sub)
    local = masked_blur(lum, np.maximum(m, 1e-3), max(1.5, scale * 0.025))
    groove = np.clip(local - lum, 0, None)
    lifted = sub * (1.0 + (groove / np.maximum(lum, 1e-3)) * 0.9 * a)[..., None]
    out = img.copy()
    out[roi] = blend(sub, lifted, m)
    return smooth_skin(out, region * a, scale, 60, 55)


def apply(img: np.ndarray, skin: np.ndarray | None, scale: float, s: SkinSettings, faces: list[Face] | None = None) -> np.ndarray:
    if skin is None:
        return img
    img = remove_blemishes(img, skin, scale, s.blemish)
    img = reduce_wrinkles(img, skin, faces or [], scale, s.wrinkles)
    img = smooth_skin(img, skin, scale, s.smooth, s.texture)
    img = even_skin_tone(img, skin, scale, s.even_tone)
    img = brighten_skin(img, skin, s.brighten)
    return img
