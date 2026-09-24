"""画像処理の共通ユーティリティ。画像は float32 BGR (0..1) を前提とする。

大きな画像 (2000 万画素超) でも実用的な速度になるよう、要素ごとの演算は
なるべく OpenCV (SIMD・マルチスレッド) の関数で行う。
"""

from __future__ import annotations

import cv2
import numpy as np


def to_float(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float32) * np.float32(1.0 / 255.0)


def to_uint8(img: np.ndarray) -> np.ndarray:
    return cv2.convertScaleAbs(img, alpha=255.0)  # 丸め・0..255 への飽和込み


def clip01(img: np.ndarray) -> np.ndarray:
    """0..1 に収める (np.clip より速い)。新しい配列を返す。"""
    out = np.maximum(img, np.float32(0.0))
    return np.minimum(out, np.float32(1.0), out=out)


_LUMA = np.array([[0.114, 0.587, 0.299]], np.float32)


def luminance(img: np.ndarray) -> np.ndarray:
    return cv2.transform(img, _LUMA)


def smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = clip01((x - np.float32(e0)) * np.float32(1.0 / (e1 - e0)))
    return t * t * (np.float32(3.0) - np.float32(2.0) * t)


def blur(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.3:
        return img
    # 大きなぼかしは縮小してから行うと桁違いに速い
    if sigma > 12 and min(img.shape[:2]) > 64:
        f = max(1, int(sigma / 6))
        h, w = img.shape[:2]
        small = cv2.resize(img, (max(1, w // f), max(1, h // f)), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), sigma / f)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.GaussianBlur(img, (0, 0), sigma)


def masked_blur(img: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    """マスク内の画素だけを使ったぼかし (正規化畳み込み)。マスク外の色が混ざらない。"""
    mask = mask.astype(np.float32, copy=False)
    m3 = mask if img.ndim == 2 else cv2.merge([mask] * img.shape[2])
    num = blur(cv2.multiply(img, m3), sigma)
    den = np.maximum(blur(mask, sigma), np.float32(1e-4))
    if img.ndim == 3:
        den = cv2.merge([den] * img.shape[2])
    return cv2.divide(num, den)


def guided_filter(guide: np.ndarray, src: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """He らの Guided Filter (グレースケールガイド版)。エッジを保ったまま平滑化する。"""
    ksize = (2 * radius + 1, 2 * radius + 1)

    def box(x):
        return cv2.boxFilter(x, -1, ksize, borderType=cv2.BORDER_REFLECT)

    guide_c = guide[..., None] if src.ndim == 3 else guide
    mean_i = box(guide)
    mean_p = box(src)
    corr_ip = box(src * guide_c)
    var_i = box(guide * guide) - mean_i * mean_i
    if src.ndim == 3:
        mean_i_c, var_i_c = mean_i[..., None], var_i[..., None]
    else:
        mean_i_c, var_i_c = mean_i, var_i
    a = (corr_ip - mean_i_c * mean_p) / (var_i_c + eps)
    b = mean_p - a * mean_i_c
    return box(a) * guide_c + box(b)


def mask_roi(mask: np.ndarray, pad: int, thresh: float = 0.01) -> tuple[slice, slice] | None:
    """マスクが有効な範囲 (+余白) の矩形。処理範囲を絞って高速化するのに使う。"""
    x, y, w, h = cv2.boundingRect((mask > thresh).view(np.uint8))
    if w == 0 or h == 0:
        return None
    H, W = mask.shape[:2]
    return slice(max(0, y - pad), min(H, y + h + pad)), slice(max(0, x - pad), min(W, x + w + pad))


def polygons_roi(shape: tuple[int, int], polygons: list[np.ndarray], pad: float) -> tuple[slice, slice] | None:
    """多角形群を囲む矩形 (+余白)。画像外なら None。"""
    pts = np.concatenate([np.asarray(p, np.float32).reshape(-1, 2) for p in polygons])
    H, W = shape
    x0, y0 = int(max(0, np.floor(pts[:, 0].min() - pad))), int(max(0, np.floor(pts[:, 1].min() - pad)))
    x1, y1 = int(min(W, np.ceil(pts[:, 0].max() + pad) + 1)), int(min(H, np.ceil(pts[:, 1].max() + pad) + 1))
    if x0 >= x1 or y0 >= y1:
        return None
    return slice(y0, y1), slice(x0, x1)


def polygon_mask_in(roi: tuple[slice, slice], polygons: list[np.ndarray], feather: float = 0.0) -> np.ndarray:
    """roi 矩形の中だけで多角形マスクを作る (画像全体を確保しないので速い)。"""
    ys, xs = roi
    mask = np.zeros((ys.stop - ys.start, xs.stop - xs.start), np.uint8)
    off = np.array([xs.start, ys.start], np.float32)
    for poly in polygons:
        cv2.fillPoly(mask, [np.round(np.asarray(poly, np.float32) - off).astype(np.int32)], 255, lineType=cv2.LINE_AA)
    out = mask.astype(np.float32) * np.float32(1.0 / 255.0)
    if feather > 0:
        out = cv2.GaussianBlur(out, (0, 0), feather)
    return out


def blend(base: np.ndarray, top: np.ndarray, alpha: np.ndarray | float) -> np.ndarray:
    """base と top を alpha で合成 (base + (top - base) * alpha)。"""
    top = top.astype(np.float32, copy=False)
    if not isinstance(alpha, np.ndarray):
        return cv2.addWeighted(base, 1.0 - float(alpha), top, float(alpha), 0.0)
    alpha = alpha.astype(np.float32, copy=False)
    if alpha.ndim == 2 and base.ndim == 3:
        alpha = cv2.merge([alpha] * base.shape[2])
    return cv2.add(base, cv2.multiply(cv2.subtract(top, base), alpha))


def bgr_to_lab(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(clip01(img), cv2.COLOR_BGR2Lab)


def lab_to_bgr(lab: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(lab.astype(np.float32, copy=False), cv2.COLOR_Lab2BGR)


def at_scale(fn, img: np.ndarray, k: float, *masks: np.ndarray) -> np.ndarray:
    """fn を縮小画像で実行し、その「変化量」だけを元解像度に戻して適用する。

    色ムラ補正など低周波の処理向け。元画像の細かい質感はそのまま残る。
    """
    if k >= 0.95:
        return fn(img, *masks)
    h, w = img.shape[:2]
    size = (max(1, round(w * k)), max(1, round(h * k)))
    small = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    small_masks = [cv2.resize(m, size, interpolation=cv2.INTER_AREA) for m in masks]
    delta = cv2.subtract(fn(small, *small_masks), small)
    return cv2.add(img, cv2.resize(delta, (w, h), interpolation=cv2.INTER_LINEAR))
