"""背景処理: 人物セグメンテーションを使った背景ぼかし。"""

from __future__ import annotations

import cv2
import numpy as np

from .ops import blend, masked_blur


def blur_background(img: np.ndarray, person: np.ndarray | None, amount: float) -> np.ndarray:
    if amount <= 0 or person is None:
        return img
    a = amount / 100.0
    h, w = img.shape[:2]
    sigma = max(h, w) * 0.012 * (0.3 + 1.7 * a)
    fg = np.clip(person, 0, 1)
    bg = 1.0 - fg
    # 背景画素だけでぼかし、人物の色が背景へにじむハロを防ぐ
    blurred = masked_blur(img, np.maximum(bg, 1e-3), sigma)
    edge = cv2.GaussianBlur(fg, (0, 0), max(1.0, max(h, w) * 0.002))
    return blend(blurred, img, edge)


def apply(img: np.ndarray, person: np.ndarray | None, blur_amount: float) -> np.ndarray:
    return blur_background(img, person, blur_amount)
