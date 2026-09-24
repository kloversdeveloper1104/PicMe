"""解析結果と設定からレタッチ済み画像を生成する処理パイプライン。"""

from __future__ import annotations

import numpy as np

from .analysis import Analysis, FaceAnalyzer
from .retouch import background, color, face, skin
from .retouch.ops import to_float, to_uint8
from .settings import RetouchSettings


def retouch(img_bgr: np.ndarray, analysis: Analysis, settings: RetouchSettings) -> np.ndarray:
    """BGR uint8 画像にレタッチを適用して BGR uint8 を返す。

    analysis は別解像度で作ったものでもよい (自動でリサイズされる)。
    """
    a = analysis.resized(img_bgr.shape[:2])
    img = to_float(img_bgr)
    scale = a.reference_scale

    # 1. 基本補正 (色・明るさ)
    img = color.apply(img, settings.color, a.skin_mask)
    # 2. 美肌
    img = skin.apply(img, a.skin_mask, scale, settings.skin)
    # 3. 顔パーツ (目・クマ・歯・唇)
    img = face.apply_color(img, a.faces, a.skin_mask, settings.face)
    # 4. シャープ (肌はなめらかさを保つため弱める)
    protect = None
    if a.skin_mask is not None and settings.skin.smooth > 0:
        protect = a.skin_mask * min(1.0, settings.skin.smooth / 60.0)
    img = color.sharpen(img, settings.color.sharpen, protect)
    # 5. 背景
    img = background.apply(img, a.person_mask, settings.background.blur)
    # 6. 変形 (小顔・デカ目) は最後に行い、他のマスクとの位置ずれを防ぐ
    img = face.apply_warp(img, a.faces, settings.face)
    return to_uint8(img)


class Processor:
    """解析器を保持して、画像 1 枚ごとの解析 → レタッチをまとめて行う。"""

    def __init__(self, analyzer: FaceAnalyzer | None = None, use_ai: bool = True):
        self.analyzer = analyzer or FaceAnalyzer(use_ai=use_ai)

    def process(self, img_bgr: np.ndarray, settings: RetouchSettings) -> tuple[np.ndarray, Analysis]:
        analysis = self.analyzer.analyze(img_bgr)
        return retouch(img_bgr, analysis, settings), analysis
