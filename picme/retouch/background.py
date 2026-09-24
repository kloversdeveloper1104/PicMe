"""背景処理: 人物セグメンテーションを使った背景ぼかし・背景の置き換え。"""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np

from ..settings import BackgroundSettings
from .ops import blend, blur, masked_blur, smoothstep, to_float


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


def parse_color(text: str) -> np.ndarray | None:
    """'#RRGGBB' / 'RRGGBB' / 'r,g,b' を BGR float (0..1) に。"""
    t = text.strip().lstrip("#")
    try:
        if "," in t:
            r, g, b = (int(v) for v in t.split(","))
        elif len(t) == 6:
            r, g, b = int(t[0:2], 16), int(t[2:4], 16), int(t[4:6], 16)
        else:
            return None
    except ValueError:
        return None
    return np.array([b, g, r], np.float32) / 255.0


@lru_cache(maxsize=4)
def _load_background(path: str) -> np.ndarray | None:
    from ..io import load_image

    try:
        return to_float(load_image(path)[0])
    except Exception:
        return None


def _cover(bg: np.ndarray, h: int, w: int) -> np.ndarray:
    """アスペクト比を保って画面いっぱいに拡大し、中央を切り出す。"""
    bh, bw = bg.shape[:2]
    k = max(w / bw, h / bh)
    resized = cv2.resize(bg, (max(w, round(bw * k)), max(h, round(bh * k))), interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_CUBIC)
    y0, x0 = (resized.shape[0] - h) // 2, (resized.shape[1] - w) // 2
    return resized[y0 : y0 + h, x0 : x0 + w]


def replace_background(img: np.ndarray, person: np.ndarray | None, s: BackgroundSettings) -> np.ndarray | None:
    """置き換えが指定されていれば合成結果を、されていなければ None を返す。"""
    if person is None:
        return None
    h, w = img.shape[:2]
    new_bg = None
    if s.replace_image:
        src = _load_background(s.replace_image)
        if src is not None:
            new_bg = _cover(src, h, w)
    if new_bg is None and s.replace_color:
        color = parse_color(s.replace_color)
        if color is not None:
            new_bg = np.broadcast_to(color, img.shape).copy()
    if new_bg is None:
        return None
    if s.blur > 0 and s.replace_image:
        new_bg = blur(new_bg, max(h, w) * 0.01 * s.blur / 100.0 * 2)

    alpha = smoothstep(0.25, 0.75, np.clip(person, 0, 1))
    alpha = cv2.GaussianBlur(alpha, (0, 0), max(0.8, max(h, w) * 0.0015))
    # 境界付近は元の背景色が混ざっているので、内側の人物の色で置き換える (フリンジ除去)
    inner = (alpha > 0.95).astype(np.float32)
    if inner.any():
        fg_color = masked_blur(img, inner + 1e-4, max(1.5, max(h, w) * 0.004))
        img = blend(img, fg_color, (1.0 - alpha) * (alpha > 0.02))
    return blend(new_bg, img, alpha)


def apply(img: np.ndarray, person: np.ndarray | None, s: BackgroundSettings) -> np.ndarray:
    replaced = replace_background(img, person, s)
    if replaced is not None:
        return replaced
    return blur_background(img, person, s.blur)
