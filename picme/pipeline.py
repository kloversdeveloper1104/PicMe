"""解析結果と設定からレタッチ済み画像を生成する処理パイプライン。"""

from __future__ import annotations

import cv2
import numpy as np

from .analysis import Analysis, Face, FaceAnalyzer
from .retouch import background, color, face, hair, heal, skin
from .retouch.ops import to_uint8
from .settings import FaceOverride, RetouchSettings


def match_override(f: Face, overrides: list[FaceOverride], shape: tuple[int, int]) -> FaceOverride | None:
    """顔の位置に最も近い人物ごとの設定を探す (顔の大きさの半分以内)。"""
    h, w = shape
    c = f.center
    best, best_d = None, 0.5 * f.scale
    for o in overrides:
        d = float(np.hypot(o.x * w - c[0], o.y * h - c[1]))
        if d < best_d:
            best, best_d = o, d
    return best


def face_plan(a: Analysis, settings: RetouchSettings) -> tuple[face.FacePlan, np.ndarray | None]:
    """顔ごとの設定と、人物ごとの美肌の強さマップを作る。"""
    plan: face.FacePlan = []
    skin_gain = None
    for f in a.faces:
        o = match_override(f, settings.face_overrides, a.shape)
        plan.append((f, o.face if o else settings.face))
        if o and o.skin != 100.0:
            if skin_gain is None:
                skin_gain = np.ones(a.shape, np.float32)
            region = np.zeros(a.shape, np.float32)
            c, r = f.center, f.scale
            cv2.ellipse(region, (int(c[0]), int(c[1])), (int(r * 0.75), int(r * 1.1)), 0, 0, 360, 1.0, -1)
            region = cv2.GaussianBlur(region, (0, 0), max(1.0, r * 0.1))
            skin_gain = skin_gain * (1.0 - region) + region * (o.skin / 100.0)
    return plan, skin_gain


def retouch(img_bgr: np.ndarray, analysis: Analysis, settings: RetouchSettings) -> np.ndarray:
    """BGR uint8 画像にレタッチを適用して BGR uint8 を返す。

    analysis は別解像度で作ったものでもよい (自動でリサイズされる)。
    """
    a = analysis.resized(img_bgr.shape[:2])
    scale = a.reference_scale
    plan, skin_gain = face_plan(a, settings)
    skin_mask = a.skin_mask
    if skin_mask is not None and skin_gain is not None:
        skin_mask = skin_mask * skin_gain

    # 1. 基本補正 (色・明るさ)
    img = color.apply(img_bgr, settings.color, a.skin_mask)  # uint8 → float32
    # 2. 美肌
    img = skin.apply(img, skin_mask, scale, settings.skin, a.faces)
    # 3. 顔パーツ (目・クマ・歯・唇) と髪
    img = face.apply_color(img, plan, a.skin_mask)
    img = hair.add_shine(img, a.hair_mask, scale, max((s.hair_shine for _, s in plan), default=settings.face.hair_shine))
    # 4. シャープ (肌はなめらかさを保つため弱める)
    protect = None
    if skin_mask is not None and settings.skin.smooth > 0:
        protect = skin_mask * min(1.0, settings.skin.smooth / 60.0)
    img = color.sharpen(img, settings.color.sharpen, protect)
    # 5. 背景
    img = background.apply(img, a.person_mask, settings.background)
    # 6. 変形 (小顔・デカ目) は最後に行い、他のマスクとの位置ずれを防ぐ
    img = face.apply_warp(img, plan)
    # 7. 修正ブラシ (ユーザーは変形後の画像を見て指定するので最後)
    img = heal.apply(img, settings.heal_spots)
    return to_uint8(img)


class Processor:
    """解析器を保持して、画像 1 枚ごとの解析 → レタッチをまとめて行う。"""

    def __init__(self, analyzer: FaceAnalyzer | None = None, use_ai: bool = True):
        self.analyzer = analyzer or FaceAnalyzer(use_ai=use_ai)

    def process(self, img_bgr: np.ndarray, settings: RetouchSettings) -> tuple[np.ndarray, Analysis]:
        analysis = self.analyzer.analyze(img_bgr)
        return retouch(img_bgr, analysis, settings), analysis
