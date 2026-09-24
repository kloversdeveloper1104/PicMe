"""画像処理の共通ユーティリティ。画像は float32 BGR (0..1) を前提とする。"""

from __future__ import annotations

import cv2
import numpy as np


def to_float(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float32) / 255.0


def to_uint8(img: np.ndarray) -> np.ndarray:
    return np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)


def luminance(img: np.ndarray) -> np.ndarray:
    b, g, r = img[..., 0], img[..., 1], img[..., 2]
    return 0.114 * b + 0.587 * g + 0.299 * r


def smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


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
    m = mask if img.ndim == 2 else mask[..., None]
    num = blur(img * m, sigma)
    den = blur(mask, sigma)
    den = den if img.ndim == 2 else den[..., None]
    return num / np.maximum(den, 1e-4)


def guided_filter(guide: np.ndarray, src: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """He らの Guided Filter (グレースケールガイド版)。エッジを保ったまま平滑化する。"""
    ksize = (2 * radius + 1, 2 * radius + 1)

    def box(x):
        return cv2.boxFilter(x, -1, ksize, borderType=cv2.BORDER_REFLECT)

    if src.ndim == 3:
        guide_c = guide[..., None]
    else:
        guide_c = guide
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
    ys, xs = np.where(mask > thresh)
    if len(ys) == 0:
        return None
    h, w = mask.shape[:2]
    y0, y1 = max(0, ys.min() - pad), min(h, ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(w, xs.max() + pad + 1)
    return slice(y0, y1), slice(x0, x1)


def blend(base: np.ndarray, top: np.ndarray, alpha: np.ndarray | float) -> np.ndarray:
    if isinstance(alpha, np.ndarray) and alpha.ndim == 2 and base.ndim == 3:
        alpha = alpha[..., None]
    return base + (top - base) * alpha


def bgr_to_lab(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.clip(img, 0, 1), cv2.COLOR_BGR2Lab)


def lab_to_bgr(lab: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
