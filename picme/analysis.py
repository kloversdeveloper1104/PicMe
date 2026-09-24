"""AI 解析: 顔ランドマーク検出と人物・肌のセグメンテーション。

MediaPipe のモデルが使えない環境でも、色ベースの肌検出にフォールバックして
処理を続けられるようにしている。
"""

from __future__ import annotations

import logging
import atexit
import threading
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import landmarks as lm
from .models import ensure_model

log = logging.getLogger(__name__)

ANALYSIS_MAX_SIDE = 1280


@dataclass
class Face:
    points: np.ndarray  # (478, 2) float32, 画像座標 (px)

    def pts(self, indices: list[int]) -> np.ndarray:
        return self.points[indices]

    @property
    def eye_distance(self) -> float:
        return float(np.linalg.norm(self.points[lm.LEFT_IRIS_CENTER] - self.points[lm.RIGHT_IRIS_CENTER]))

    @property
    def scale(self) -> float:
        """顔の大きさの目安 (px)。処理半径などはこれに比例させる。"""
        oval = self.pts(lm.FACE_OVAL)
        w = oval[:, 0].max() - oval[:, 0].min()
        return float(max(w, self.eye_distance * 2.0, 1.0))

    @property
    def center(self) -> np.ndarray:
        return self.pts(lm.FACE_OVAL).mean(axis=0)

    def scaled(self, sx: float, sy: float) -> "Face":
        return Face(self.points * np.array([sx, sy], dtype=np.float32))


@dataclass
class Analysis:
    shape: tuple[int, int]  # (h, w)
    faces: list[Face] = field(default_factory=list)
    skin_mask: np.ndarray | None = None  # float32 0..1 (目・眉・唇は除外済み)
    person_mask: np.ndarray | None = None  # float32 0..1 (前景=人物)
    hair_mask: np.ndarray | None = None  # float32 0..1

    @property
    def reference_scale(self) -> float:
        """処理半径の基準。顔があれば最大の顔、なければ画像サイズから決める。"""
        if self.faces:
            return max(f.scale for f in self.faces)
        return max(self.shape) * 0.25

    def resized(self, shape: tuple[int, int]) -> "Analysis":
        """別解像度 (プレビュー ↔ 書き出し) の画像向けに座標とマスクを変換する。"""
        h, w = shape
        if (h, w) == tuple(self.shape):
            return self
        sx, sy = w / self.shape[1], h / self.shape[0]

        def rs(m):
            return None if m is None else cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)

        return Analysis(
            shape=(h, w),
            faces=[f.scaled(sx, sy) for f in self.faces],
            skin_mask=rs(self.skin_mask),
            person_mask=rs(self.person_mask),
            hair_mask=rs(self.hair_mask),
        )


def polygon_mask(shape: tuple[int, int], polygons: list[np.ndarray], feather: float = 0.0) -> np.ndarray:
    """多角形を塗りつぶした float32 マスク。feather>0 でエッジをぼかす。"""
    mask = np.zeros(shape, np.uint8)
    for poly in polygons:
        cv2.fillPoly(mask, [np.round(poly).astype(np.int32)], 255, lineType=cv2.LINE_AA)
    out = mask.astype(np.float32) / 255.0
    if feather > 0:
        out = cv2.GaussianBlur(out, (0, 0), feather)
    return out


def color_skin_mask(img: np.ndarray) -> np.ndarray:
    """YCrCb 色空間による簡易肌検出 (AI モデルが無いときのフォールバック)。"""
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(ycrcb, (40, 135, 85), (255, 180, 135))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return mask.astype(np.float32) / 255.0


class FaceAnalyzer:
    """顔ランドマーク + 人物/肌セグメンテーション。

    MediaPipe のモデル生成は重いので、インスタンスを使い回すこと。
    スレッドセーフにするため推論はロックで直列化する。
    """

    def __init__(self, use_ai: bool = True, max_faces: int = 10, download: bool = True, detect_small_faces: bool = True):
        self._lock = threading.Lock()
        self.detect_small_faces = detect_small_faces
        self._landmarker = None
        self._segmenter = None
        self.ai_available = False
        if use_ai:
            self._init_models(max_faces, download)

    def _init_models(self, max_faces: int, download: bool) -> None:
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions, vision
        except Exception as exc:
            log.warning("mediapipe を読み込めません (AI 機能なしで続行): %s", exc)
            return
        self._mp = mp
        if path := ensure_model("face_landmarker.task", download):
            try:
                self._landmarker = vision.FaceLandmarker.create_from_options(
                    vision.FaceLandmarkerOptions(
                        base_options=BaseOptions(model_asset_path=str(path)),
                        running_mode=vision.RunningMode.IMAGE,
                        num_faces=max_faces,
                        min_face_detection_confidence=0.4,
                        min_face_presence_confidence=0.4,
                    )
                )
            except Exception as exc:
                log.warning("顔ランドマークモデルの初期化に失敗: %s", exc)
        if path := ensure_model("selfie_multiclass_256x256.tflite", download):
            try:
                self._segmenter = vision.ImageSegmenter.create_from_options(
                    vision.ImageSegmenterOptions(
                        base_options=BaseOptions(model_asset_path=str(path)),
                        running_mode=vision.RunningMode.IMAGE,
                        output_confidence_masks=True,
                        output_category_mask=False,
                    )
                )
            except Exception as exc:
                log.warning("セグメンテーションモデルの初期化に失敗: %s", exc)
        self.ai_available = self._landmarker is not None
        # MediaPipe は内部のスレッドプールが止まった後に close されると例外を出すため、
        # それより前 (スレッド終了フック) に閉じておく
        getattr(threading, "_register_atexit", atexit.register)(self.close)

    def close(self) -> None:
        for model in (self._landmarker, self._segmenter):
            if model is not None:
                try:
                    model.close()
                except Exception:
                    pass
        self._landmarker = self._segmenter = None

    # ------------------------------------------------------------------
    def analyze(self, img: np.ndarray) -> Analysis:
        """BGR uint8 画像を解析する。大きい画像は縮小して推論し、結果を元サイズに戻す。"""
        h, w = img.shape[:2]
        k = min(1.0, ANALYSIS_MAX_SIDE / max(h, w))
        small = cv2.resize(img, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA) if k < 1 else img
        sh, sw = small.shape[:2]

        faces: list[Face] = []
        skin = person = hair = None
        rgb = np.ascontiguousarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))

        with self._lock:
            if self._landmarker is not None:
                faces = self._detect_faces(img, k)
            if self._segmenter is not None:
                mp_img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
                seg = self._segmenter.segment(mp_img)
                conf = [np.array(m.numpy_view(), np.float32).reshape(sh, sw) for m in seg.confidence_masks]
                # 0:背景 1:髪 2:体の肌 3:顔の肌 4:服 5:その他(アクセサリー等)
                if len(conf) >= 4:
                    person = np.clip(1.0 - conf[0], 0, 1)
                    hair = np.clip(conf[1], 0, 1)
                    skin = np.clip(conf[2] + conf[3], 0, 1)
                    # 自信度の低い領域 (肌色の壁・木材など) を落とし、人物内に限定する
                    t = np.clip((skin - 0.25) / 0.5, 0, 1)
                    skin = t * t * (3 - 2 * t) * np.clip((person - 0.2) / 0.4, 0, 1)

        if skin is None:
            skin = color_skin_mask(small)
            if faces:  # 顔がある場合は顔周辺に限定して誤検出を減らす
                region = np.zeros((sh, sw), np.float32)
                for f in faces:
                    c, r = f.center, f.scale
                    cv2.ellipse(region, (int(c[0]), int(c[1])), (int(r * 0.9), int(r * 1.4)), 0, 0, 360, 1.0, -1)
                skin *= region

        analysis = Analysis(shape=(sh, sw), faces=faces, skin_mask=skin, person_mask=person, hair_mask=hair)
        analysis.skin_mask = self._refine_skin(analysis)
        return analysis.resized((h, w))

    def _landmarks(self, rgb: np.ndarray) -> list[np.ndarray]:
        h, w = rgb.shape[:2]
        mp_img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self._landmarker.detect(mp_img)
        out = []
        for face_lms in result.face_landmarks:
            pts = np.array([[p.x * w, p.y * h] for p in face_lms], np.float32)
            if len(pts) >= 478:
                out.append(pts)
        return out

    def _detect_faces(self, img: np.ndarray, k: float) -> list[Face]:
        """画像全体 + 重なりのあるタイルで顔を検出する (集合写真の小さな顔も拾う)。

        戻り値の座標は解析用に k 倍縮小した画像の座標系。
        """
        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        def run(x0: int, y0: int, x1: int, y1: int) -> list[Face]:
            tile = rgb[y0:y1, x0:x1]
            f = min(ANALYSIS_MAX_SIDE, 960 if (x1 - x0) < w else ANALYSIS_MAX_SIDE) / max(tile.shape[:2])
            f = min(f, 3.0)
            if abs(f - 1.0) > 1e-3:
                interp = cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC
                tile = cv2.resize(tile, (max(1, round((x1 - x0) * f)), max(1, round((y1 - y0) * f))), interpolation=interp)
            return [Face((pts / f + np.array([x0, y0], np.float32)) * k) for pts in self._landmarks(tile)]

        faces = run(0, 0, w, h)
        if self.detect_small_faces:
            for grid in (2, 3):
                tw, th = int(w / grid * 1.5), int(h / grid * 1.5)
                for gy in range(grid):
                    for gx in range(grid):
                        x0 = min(max(0, int(gx * w / grid - (tw - w / grid) / 2)), max(0, w - tw))
                        y0 = min(max(0, int(gy * h / grid - (th - h / grid) / 2)), max(0, h - th))
                        for cand in run(x0, y0, min(w, x0 + tw), min(h, y0 + th)):
                            # 既に見つかっている顔と重なるものは捨てる
                            if all(np.linalg.norm(cand.center - f.center) > 0.4 * max(cand.scale, f.scale) for f in faces):
                                faces.append(cand)
        return faces

    @staticmethod
    def _refine_skin(a: Analysis) -> np.ndarray:
        """肌マスクから目・眉・唇を除外し、エッジを整える。"""
        skin = a.skin_mask.copy()
        for f in a.faces:
            s = f.scale
            excl = []
            for idx in (lm.LEFT_EYE, lm.RIGHT_EYE, lm.LEFT_BROW, lm.RIGHT_BROW, lm.LIPS_OUTER):
                pts = cv2.convexHull(f.pts(idx)).reshape(-1, 2)
                c = pts.mean(axis=0)
                excl.append(c + (pts - c) * 1.25)  # 少し広げて境界の滲みを防ぐ
            m = polygon_mask(a.shape, excl, feather=max(1.0, s * 0.01))
            skin *= 1.0 - m
        feather = max(1.0, a.reference_scale * 0.008)
        return cv2.GaussianBlur(skin, (0, 0), feather)
