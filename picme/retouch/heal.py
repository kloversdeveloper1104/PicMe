"""修正ブラシ (スポット修復)。指定した円の中を、周囲の似た肌の質感で置き換える。"""

from __future__ import annotations

import cv2
import numpy as np

from .ops import blend, masked_blur


def _disk(shape: tuple[int, int], cx: float, cy: float, r: float) -> np.ndarray:
    ys, xs = np.mgrid[0 : shape[0], 0 : shape[1]].astype(np.float32)
    d = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    return np.clip((r - d) / max(1.0, r * 0.35) + 0.5, 0, 1)


def heal_spot(img: np.ndarray, cx: float, cy: float, r: float) -> np.ndarray:
    h, w = img.shape[:2]
    r = max(2.0, r)
    pad = int(r * 4) + 2
    x0, y0 = max(0, int(cx) - pad), max(0, int(cy) - pad)
    x1, y1 = min(w, int(cx) + pad + 1), min(h, int(cy) + pad + 1)
    sub = img[y0:y1, x0:x1]
    lcx, lcy = cx - x0, cy - y0
    shape = sub.shape[:2]
    target = _disk(shape, lcx, lcy, r)
    ring = _disk(shape, lcx, lcy, r * 1.6) * (1.0 - _disk(shape, lcx, lcy, r * 1.05))

    # 周囲で一番「リング部分」が似ている場所を質感のコピー元にする
    best, best_cost = None, np.inf
    for ang in np.linspace(0, 2 * np.pi, 12, endpoint=False):
        dx, dy = np.cos(ang) * r * 2.2, np.sin(ang) * r * 2.2
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted = cv2.warpAffine(sub, M, (shape[1], shape[0]), borderMode=cv2.BORDER_REFLECT)
        sx, sy = lcx + dx, lcy + dy
        if not (r * 1.6 <= sx + x0 < w - r * 1.6 and r * 1.6 <= sy + y0 < h - r * 1.6):
            continue
        cost = float((((shifted - sub) ** 2).sum(axis=2) * ring).sum())
        if cost < best_cost:
            best, best_cost = shifted, cost
    if best is None:
        best = sub

    # 低周波 (色・明るさ) は周囲から、高周波 (肌理) はコピー元から
    low = masked_blur(sub, 1.0 - np.clip(target * 1.3, 0, 1) + 1e-3, max(1.0, r * 0.8))
    sigma = max(0.8, r * 0.3)
    detail = best - cv2.GaussianBlur(best, (0, 0), sigma)
    fill = cv2.GaussianBlur(low, (0, 0), sigma) + detail
    out = img.copy()
    out[y0:y1, x0:x1] = blend(sub, fill, target)
    return out


def apply(img: np.ndarray, spots: list[list[float]]) -> np.ndarray:
    """spots: [x, y, r] (x, y は 0..1、r は長辺に対する割合)。"""
    if not spots:
        return img
    h, w = img.shape[:2]
    side = max(h, w)
    for x, y, r in spots:
        img = heal_spot(img, x * w, y * h, r * side)
    return img
