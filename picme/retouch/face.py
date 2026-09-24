"""顔パーツ補正: 目・クマ・歯・唇、および輪郭/目の変形 (リキッド)。"""

from __future__ import annotations

import cv2
import numpy as np

from .. import landmarks as lm
from ..analysis import Face, polygon_mask
from ..settings import FaceSettings
from .ops import bgr_to_lab, blend, blur, lab_to_bgr, luminance, mask_roi, smoothstep


def _eye_width(face: Face, idx: list[int]) -> float:
    pts = face.pts(idx)
    return float(np.linalg.norm(pts[0] - pts[8]))


def _expand(pts: np.ndarray, k: float) -> np.ndarray:
    c = pts.mean(axis=0)
    return c + (pts - c) * k


# ---------------------------------------------------------------------------
# 色・明るさ系
# ---------------------------------------------------------------------------
def brighten_eyes(img: np.ndarray, faces: list[Face], amount: float) -> np.ndarray:
    if amount <= 0 or not faces:
        return img
    a = amount / 100.0
    for f in faces:
        for idx in (lm.LEFT_EYE, lm.RIGHT_EYE):
            ew = _eye_width(f, idx)
            mask = polygon_mask(img.shape[:2], [_expand(f.pts(idx), 1.05)], feather=max(0.8, ew * 0.06))
            roi = mask_roi(mask, int(ew * 0.3) + 2)
            if roi is None:
                continue
            sub, m = img[roi], mask[roi]
            # クリア感 (局所コントラスト) + 明るさ + 白目の充血を抑える
            detail = sub - blur(sub, max(1.0, ew * 0.08))
            bright = sub + detail * (0.8 * a)
            bright = 1.0 - np.power(np.clip(1.0 - bright, 0, 1), 1.0 + 0.5 * a)
            lab = bgr_to_lab(bright)
            whites = smoothstep(55, 80, lab[..., 0])
            lab[..., 1] *= 1.0 - 0.6 * a * whites
            lab[..., 2] *= 1.0 - 0.4 * a * whites
            img = img.copy()
            img[roi] = blend(sub, lab_to_bgr(lab), m * min(1.0, 0.4 + a))
    return img


def _under_eye_polygon(f: Face, lower_idx: list[int], ew: float, depth: float) -> np.ndarray:
    lower = f.pts(lower_idx)
    n = len(lower)
    t = np.linspace(0, np.pi, n)
    offset = (0.08 + depth * np.sin(t)) * ew
    shifted = lower + np.stack([np.zeros(n), offset], axis=1).astype(np.float32)
    top = lower + np.stack([np.zeros(n), np.full(n, 0.05 * ew)], axis=1).astype(np.float32)
    return np.concatenate([top, shifted[::-1]], axis=0)


def remove_dark_circles(img: np.ndarray, faces: list[Face], skin: np.ndarray | None, amount: float) -> np.ndarray:
    """目の下の暗さ・青み/紫みを、すぐ下の頬の色に近づける。"""
    if amount <= 0 or not faces:
        return img
    a = amount / 100.0
    shape = img.shape[:2]
    for f in faces:
        for idx, lower_idx in ((lm.LEFT_EYE, lm.LEFT_EYE_LOWER), (lm.RIGHT_EYE, lm.RIGHT_EYE_LOWER)):
            ew = _eye_width(f, idx)
            region = _under_eye_polygon(f, lower_idx, ew, 0.45)
            mask = polygon_mask(shape, [region], feather=max(1.0, ew * 0.12))
            eye = polygon_mask(shape, [_expand(f.pts(idx), 1.2)], feather=max(0.8, ew * 0.04))
            mask *= 1.0 - eye
            if skin is not None:
                mask *= np.clip(skin * 1.5, 0, 1)
            cheek = polygon_mask(shape, [region + np.array([0, ew * 0.55], np.float32)])
            roi = mask_roi(mask + cheek, int(ew * 0.4) + 2)
            if roi is None:
                continue
            sub, m, c = img[roi], mask[roi], cheek[roi]
            if c.sum() < 4:
                continue
            lab = bgr_to_lab(sub)
            ref = (lab * c[..., None]).reshape(-1, 3).sum(0) / c.sum()
            local = blur(lab, max(1.0, ew * 0.1))
            delta = ref[None, None, :] - local
            delta[..., 0] = np.maximum(delta[..., 0], 0)  # 明るくする方向だけ
            lab = lab + delta * (0.85 * a)
            img = img.copy()
            img[roi] = blend(sub, lab_to_bgr(lab), m)
    return img


def whiten_teeth(img: np.ndarray, faces: list[Face], amount: float) -> np.ndarray:
    if amount <= 0 or not faces:
        return img
    a = amount / 100.0
    for f in faces:
        inner = f.pts(lm.LIPS_INNER)
        area = cv2.contourArea(inner.astype(np.float32))
        if area < (f.scale * 0.02) ** 2:  # 口を閉じている
            continue
        mouth = polygon_mask(img.shape[:2], [inner], feather=max(0.8, f.scale * 0.004))
        roi = mask_roi(mouth, 4)
        if roi is None:
            continue
        sub, m = img[roi], mouth[roi]
        lab = bgr_to_lab(sub)
        L, A, B = lab[..., 0], lab[..., 1], lab[..., 2]
        chroma = np.sqrt(A * A + B * B)
        teeth = smoothstep(30, 55, L) * (1.0 - smoothstep(30, 50, chroma))
        teeth = cv2.GaussianBlur(teeth, (0, 0), max(0.6, f.scale * 0.003))
        lab[..., 2] = np.where(B > 0, B * (1.0 - 0.75 * a), B)
        lab[..., 1] = A * (1.0 - 0.4 * a)
        lab[..., 0] = L + (100.0 - L) * 0.22 * a
        img = img.copy()
        img[roi] = blend(sub, lab_to_bgr(lab), m * teeth)
    return img


def enhance_lips(img: np.ndarray, faces: list[Face], amount: float) -> np.ndarray:
    if amount <= 0 or not faces:
        return img
    a = amount / 100.0
    for f in faces:
        outer = polygon_mask(img.shape[:2], [f.pts(lm.LIPS_OUTER)], feather=max(0.8, f.scale * 0.012))
        inner = polygon_mask(img.shape[:2], [f.pts(lm.LIPS_INNER)], feather=max(0.8, f.scale * 0.006))
        mask = np.clip(outer - inner, 0, 1) * 0.8
        roi = mask_roi(mask, 4)
        if roi is None:
            continue
        sub, m = img[roi], mask[roi]
        lab = bgr_to_lab(sub)
        lab[..., 1] = lab[..., 1] * (1.0 + 0.3 * a) + 3.0 * a
        lab[..., 2] = lab[..., 2] * (1.0 + 0.1 * a)
        img = img.copy()
        img[roi] = blend(sub, lab_to_bgr(lab), m)
    return img


# ---------------------------------------------------------------------------
# 変形 (リキッド) 系
# ---------------------------------------------------------------------------
class WarpField:
    """複数の局所変形を合成し、最後に 1 回だけ remap する。

    push による変位は重み付き平均で合成するため、隣り合う制御点が重なっても
    変位量が積み上がらず、画像が折り返す (歪む) ことがない。
    """

    def __init__(self, shape: tuple[int, int]):
        self.h, self.w = shape
        self.dx = np.zeros(shape, np.float32)
        self.dy = np.zeros(shape, np.float32)
        self.px = np.zeros(shape, np.float32)
        self.py = np.zeros(shape, np.float32)
        self.pw = np.zeros(shape, np.float32)
        self.active = False

    def _window(self, c: np.ndarray, r: float):
        x0, x1 = max(0, int(c[0] - r)), min(self.w, int(c[0] + r) + 2)
        y0, y1 = max(0, int(c[1] - r)), min(self.h, int(c[1] + r) + 2)
        if x0 >= x1 or y0 >= y1:
            return None
        ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        return (slice(y0, y1), slice(x0, x1)), xs, ys

    def push(self, center: np.ndarray, move: np.ndarray, radius: float) -> None:
        """center 付近の画素を move 方向へ滑らかに押し出す。"""
        win = self._window(center, radius)
        if win is None:
            return
        sl, xs, ys = win
        d2 = ((xs - center[0]) ** 2 + (ys - center[1]) ** 2) / (radius * radius)
        wgt = np.where(d2 < 1.0, (1.0 - d2) ** 2, 0.0).astype(np.float32)
        self.px[sl] += wgt * move[0]
        self.py[sl] += wgt * move[1]
        self.pw[sl] += wgt
        self.active = True

    def magnify(self, center: np.ndarray, radius: float, strength: float) -> None:
        """center を中心に円形に拡大する (デカ目)。"""
        win = self._window(center, radius)
        if win is None:
            return
        sl, xs, ys = win
        vx, vy = xs - center[0], ys - center[1]
        d2 = (vx * vx + vy * vy) / (radius * radius)
        k = np.where(d2 < 1.0, strength * (1.0 - d2) ** 2, 0.0).astype(np.float32)
        self.dx[sl] -= vx * k
        self.dy[sl] -= vy * k
        self.active = True

    def displacement(self) -> tuple[np.ndarray, np.ndarray]:
        norm = np.maximum(self.pw, 1.0)
        return self.dx - self.px / norm, self.dy - self.py / norm

    def apply(self, img: np.ndarray) -> np.ndarray:
        if not self.active:
            return img
        dx, dy = self.displacement()
        gx, gy = np.meshgrid(np.arange(self.w, dtype=np.float32), np.arange(self.h, dtype=np.float32))
        return cv2.remap(img, gx + dx, gy + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def build_warp(shape: tuple[int, int], faces: list[Face], s: FaceSettings) -> WarpField:
    field = WarpField(shape)
    for f in faces:
        if s.slim > 0:
            a = s.slim / 100.0
            # 顔の中心軸 (額 → 顎) に垂直な方向へ、輪郭を水平に内側へ寄せる
            top, chin = f.points[lm.FOREHEAD], f.points[lm.CHIN]
            axis = chin - top
            axis = axis / max(float(np.linalg.norm(axis)), 1e-3)
            for side in (lm.JAW_RIGHT, lm.JAW_LEFT):
                n = len(side)
                for i, idx in enumerate(side):
                    p = f.points[idx]
                    rel = p - top
                    inward = -(rel - axis * float(rel @ axis))  # 軸へ向かう成分
                    dist = float(np.linalg.norm(inward))
                    if dist < 1:
                        continue
                    # 頬の上部と顎先は弱め、エラ付近を最も強く
                    w = np.sin(np.pi * (i + 1) / (n + 1)) ** 1.5
                    move = inward / dist * (f.scale * 0.04 * a * w)
                    field.push(p, move.astype(np.float32), f.scale * 0.16)
        if s.eye_enlarge > 0:
            a = s.eye_enlarge / 100.0
            for idx, iris in ((lm.LEFT_EYE, lm.LEFT_IRIS_CENTER), (lm.RIGHT_EYE, lm.RIGHT_IRIS_CENTER)):
                ew = _eye_width(f, idx)
                center = (f.pts(idx).mean(axis=0) + f.points[iris]) / 2
                field.magnify(center, ew * 1.1, 0.25 * a)
    return field


def apply_color(img: np.ndarray, faces: list[Face], skin: np.ndarray | None, s: FaceSettings) -> np.ndarray:
    img = remove_dark_circles(img, faces, skin, s.dark_circles)
    img = brighten_eyes(img, faces, s.eye_brighten)
    img = whiten_teeth(img, faces, s.teeth_whiten)
    img = enhance_lips(img, faces, s.lip_color)
    return img


def apply_warp(img: np.ndarray, faces: list[Face], s: FaceSettings) -> np.ndarray:
    if not faces or (s.slim <= 0 and s.eye_enlarge <= 0):
        return img
    return build_warp(img.shape[:2], faces, s).apply(img)
