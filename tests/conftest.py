import os

import numpy as np
import pytest

# テストはネットワークに依存させない (モデルが既にあれば使う)
os.environ.setdefault("PICME_OFFLINE", "1")


@pytest.fixture
def portrait_like() -> np.ndarray:
    """肌色の楕円と背景からなる合成画像 (BGR uint8)。"""
    import cv2

    rng = np.random.default_rng(0)
    img = np.full((360, 300, 3), (90, 120, 60), np.uint8)
    cv2.ellipse(img, (150, 180), (90, 120), 0, 0, 360, (140, 170, 220), -1)
    noise = rng.normal(0, 6, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)
