"""髪のツヤ出し。"""

from __future__ import annotations

import numpy as np

from .ops import blend, blur, luminance, mask_roi, smoothstep


def add_shine(img: np.ndarray, hair: np.ndarray | None, scale: float, amount: float) -> np.ndarray:
    """髪の領域で局所コントラストとハイライトを強め、ツヤ感を出す。"""
    if amount <= 0 or hair is None or hair.max() < 0.1:
        return img
    a = amount / 100.0
    roi = mask_roi(hair, int(scale * 0.05) + 2, 0.05)
    if roi is None:
        return img
    out = img.copy()
    out[roi] = _shine(img[roi], hair[roi], scale, a)
    return out


def _shine(img: np.ndarray, hair: np.ndarray, scale: float, a: float) -> np.ndarray:
    lum = luminance(img)
    local = blur(lum, max(1.5, scale * 0.02))
    detail = lum - local
    # 周囲より明るい部分 (光の当たる毛束) をさらに明るく
    highlight = np.clip(detail, 0, None) * (1.5 * a)
    gain = 1.0 + 0.25 * a * smoothstep(0.15, 0.7, local)
    out = img * gain[..., None] + (detail * 0.6 * a + highlight)[..., None]
    return blend(img, out, np.clip(hair, 0, 1) * 0.9)
