"""顔パーツ補正: 目・クマ・歯・唇、および輪郭/目の変形 (リキッド)。"""

from __future__ import annotations

import cv2
import numpy as np

from .. import landmarks as lm
from ..analysis import Face
from ..settings import FaceSettings
from .ops import bgr_to_lab, blend, blur, clip01, lab_to_bgr, polygon_mask_in, polygons_roi, smoothstep


def _eye_width(face: Face, idx: list[int]) -> float:
    pts = face.pts(idx)
    return float(np.linalg.norm(pts[0] - pts[8]))


def _expand(pts: np.ndarray, k: float) -> np.ndarray:
    c = pts.mean(axis=0)
    return c + (pts - c) * k


# ---------------------------------------------------------------------------
# 色・明るさ系 (いずれも顔パーツ周辺の矩形だけを処理し、img を直接書き換える)
# ---------------------------------------------------------------------------
def brighten_eyes(img: np.ndarray, f: Face, amount: float) -> None:
    if amount <= 0:
        return
    a = amount / 100.0
    for idx in (lm.LEFT_EYE, lm.RIGHT_EYE):
        ew = _eye_width(f, idx)
        poly = _expand(f.pts(idx), 1.05)
        roi = polygons_roi(img.shape[:2], [poly], ew * 0.3 + 2)
        if roi is None:
            continue
        m = polygon_mask_in(roi, [poly], feather=max(0.8, ew * 0.06))
        sub = img[roi]
        # クリア感 (局所コントラスト) + 明るさ + 白目の充血を抑える
        detail = sub - blur(sub, max(1.0, ew * 0.08))
        bright = sub + detail * (0.8 * a)
        bright = 1.0 - np.power(clip01(1.0 - bright), 1.0 + 0.5 * a)
        lab = bgr_to_lab(bright)
        whites = smoothstep(55, 80, lab[..., 0])
        lab[..., 1] *= 1.0 - 0.6 * a * whites
        lab[..., 2] *= 1.0 - 0.4 * a * whites
        img[roi] = blend(sub, lab_to_bgr(lab), m * min(1.0, 0.4 + a))


def _under_eye_polygon(f: Face, lower_idx: list[int], ew: float, depth: float) -> np.ndarray:
    lower = f.pts(lower_idx)
    n = len(lower)
    t = np.linspace(0, np.pi, n)
    offset = (0.08 + depth * np.sin(t)) * ew
    shifted = lower + np.stack([np.zeros(n), offset], axis=1).astype(np.float32)
    top = lower + np.stack([np.zeros(n), np.full(n, 0.05 * ew)], axis=1).astype(np.float32)
    return np.concatenate([top, shifted[::-1]], axis=0)


def remove_dark_circles(img: np.ndarray, f: Face, skin: np.ndarray | None, amount: float) -> None:
    """目の下の暗さ・青み/紫みを、すぐ下の頬の色に近づける。"""
    if amount <= 0:
        return
    a = amount / 100.0
    for idx, lower_idx in ((lm.LEFT_EYE, lm.LEFT_EYE_LOWER), (lm.RIGHT_EYE, lm.RIGHT_EYE_LOWER)):
        ew = _eye_width(f, idx)
        region = _under_eye_polygon(f, lower_idx, ew, 0.45)
        cheek_poly = region + np.array([0, ew * 0.55], np.float32)
        eye_poly = _expand(f.pts(idx), 1.2)
        roi = polygons_roi(img.shape[:2], [region, cheek_poly, eye_poly], ew * 0.4 + 2)
        if roi is None:
            continue
        m = polygon_mask_in(roi, [region], feather=max(1.0, ew * 0.12))
        m *= 1.0 - polygon_mask_in(roi, [eye_poly], feather=max(0.8, ew * 0.04))
        if skin is not None:
            m *= clip01(skin[roi] * 1.5)
        c = polygon_mask_in(roi, [cheek_poly])
        if c.sum() < 4:
            continue
        sub = img[roi]
        lab = bgr_to_lab(sub)
        ref = (lab * c[..., None]).reshape(-1, 3).sum(0) / c.sum()
        local = blur(lab, max(1.0, ew * 0.1))
        delta = ref[None, None, :].astype(np.float32) - local
        delta[..., 0] = np.maximum(delta[..., 0], 0)  # 明るくする方向だけ
        lab = lab + delta * np.float32(0.85 * a)
        img[roi] = blend(sub, lab_to_bgr(lab), m)


def whiten_teeth(img: np.ndarray, f: Face, amount: float) -> None:
    if amount <= 0:
        return
    a = amount / 100.0
    inner = f.pts(lm.LIPS_INNER)
    if cv2.contourArea(inner.astype(np.float32)) < (f.scale * 0.02) ** 2:  # 口を閉じている
        return
    roi = polygons_roi(img.shape[:2], [inner], 4 + f.scale * 0.01)
    if roi is None:
        return
    m = polygon_mask_in(roi, [inner], feather=max(0.8, f.scale * 0.004))
    sub = img[roi]
    lab = bgr_to_lab(sub)
    L, A, B = lab[..., 0], lab[..., 1], lab[..., 2]
    chroma = np.sqrt(A * A + B * B)
    teeth = smoothstep(30, 55, L) * (1.0 - smoothstep(30, 50, chroma))
    teeth = cv2.GaussianBlur(teeth, (0, 0), max(0.6, f.scale * 0.003))
    lab[..., 2] = np.where(B > 0, B * (1.0 - 0.75 * a), B)
    lab[..., 1] = A * (1.0 - 0.4 * a)
    lab[..., 0] = L + (100.0 - L) * 0.22 * a
    img[roi] = blend(sub, lab_to_bgr(lab), m * teeth)


def enhance_lips(img: np.ndarray, f: Face, amount: float) -> None:
    if amount <= 0:
        return
    a = amount / 100.0
    outer, inner = f.pts(lm.LIPS_OUTER), f.pts(lm.LIPS_INNER)
    roi = polygons_roi(img.shape[:2], [outer], 4 + f.scale * 0.04)
    if roi is None:
        return
    m = polygon_mask_in(roi, [outer], feather=max(0.8, f.scale * 0.012))
    m = clip01(m - polygon_mask_in(roi, [inner], feather=max(0.8, f.scale * 0.006))) * 0.8
    sub = img[roi]
    lab = bgr_to_lab(sub)
    lab[..., 1] = lab[..., 1] * (1.0 + 0.3 * a) + 3.0 * a
    lab[..., 2] = lab[..., 2] * (1.0 + 0.1 * a)
    img[roi] = blend(sub, lab_to_bgr(lab), m)


# ---------------------------------------------------------------------------
# 変形 (リキッド) 系
# ---------------------------------------------------------------------------
class WarpField:
    """複数の局所変形を合成し、最後に 1 回だけ remap する。

    push による変位は重み付き平均で合成するため、隣り合う制御点が重なっても
    変位量が積み上がらず、画像が折り返す (歪む) ことがない。
    変位場は変形がかかる範囲の矩形だけに確保する (大きな画像でも省メモリ)。
    """

    def __init__(self, shape: tuple[int, int]):
        self.h, self.w = shape
        self._ops: list[tuple] = []
        self.active = False

    def push(self, center: np.ndarray, move: np.ndarray, radius: float) -> None:
        """center 付近の画素を move 方向へ滑らかに押し出す。"""
        self._ops.append(("push", np.asarray(center, np.float32), np.asarray(move, np.float32), float(radius)))
        self.active = True

    def magnify(self, center: np.ndarray, radius: float, strength: float) -> None:
        """center を中心に円形に拡大する (デカ目)。"""
        self._ops.append(("magnify", np.asarray(center, np.float32), float(strength), float(radius)))
        self.active = True

    def bounds(self) -> tuple[slice, slice] | None:
        x0 = y0 = np.inf
        x1 = y1 = -np.inf
        for _, c, _, r in self._ops:
            x0, y0 = min(x0, c[0] - r), min(y0, c[1] - r)
            x1, y1 = max(x1, c[0] + r), max(y1, c[1] + r)
        x0, y0 = max(0, int(np.floor(x0))), max(0, int(np.floor(y0)))
        x1, y1 = min(self.w, int(np.ceil(x1)) + 1), min(self.h, int(np.ceil(y1)) + 1)
        if x0 >= x1 or y0 >= y1:
            return None
        return slice(y0, y1), slice(x0, x1)

    def displacement(self, roi: tuple[slice, slice]) -> tuple[np.ndarray, np.ndarray]:
        """roi 内の各出力画素について「どこから画素を持ってくるか」の変位。"""
        oy, ox = roi[0].start, roi[1].start
        shape = (roi[0].stop - oy, roi[1].stop - ox)
        dx, dy = np.zeros(shape, np.float32), np.zeros(shape, np.float32)
        px, py, pw = np.zeros(shape, np.float32), np.zeros(shape, np.float32), np.zeros(shape, np.float32)
        for kind, c, param, r in self._ops:
            # 各変形は自分の影響半径の中だけ計算する
            x0, x1 = max(ox, int(c[0] - r)), min(roi[1].stop, int(c[0] + r) + 2)
            y0, y1 = max(oy, int(c[1] - r)), min(roi[0].stop, int(c[1] + r) + 2)
            if x0 >= x1 or y0 >= y1:
                continue
            ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32)
            vx, vy = xs - c[0], ys - c[1]
            d2 = (vx * vx + vy * vy) / (r * r)
            fall = np.where(d2 < 1.0, (1.0 - d2) ** 2, 0.0).astype(np.float32)
            sl = (slice(y0 - oy, y1 - oy), slice(x0 - ox, x1 - ox))
            if kind == "push":
                px[sl] += fall * param[0]
                py[sl] += fall * param[1]
                pw[sl] += fall
            else:
                dx[sl] -= vx * fall * param
                dy[sl] -= vy * fall * param
        norm = np.maximum(pw, 1.0)
        return dx - px / norm, dy - py / norm

    def apply(self, img: np.ndarray) -> np.ndarray:
        if not self.active:
            return img
        roi = self.bounds()
        if roi is None:
            return img
        dx, dy = self.displacement(roi)
        ys, xs = np.mgrid[roi[0], roi[1]].astype(np.float32)
        out = img.copy()
        out[roi] = cv2.remap(img, xs + dx, ys + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return out


FacePlan = list[tuple[Face, FaceSettings]]


def build_warp(shape: tuple[int, int], plan: FacePlan) -> WarpField:
    field = WarpField(shape)
    for f, s in plan:
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


def apply_color(img: np.ndarray, plan: FacePlan, skin: np.ndarray | None) -> np.ndarray:
    """顔ごとに (人物ごとの設定を反映した) 色・明るさ補正を行う。"""
    if not any(s.dark_circles or s.eye_brighten or s.teeth_whiten or s.lip_color for _, s in plan):
        return img
    img = img.copy()
    for f, s in plan:
        remove_dark_circles(img, f, skin, s.dark_circles)
        brighten_eyes(img, f, s.eye_brighten)
        whiten_teeth(img, f, s.teeth_whiten)
        enhance_lips(img, f, s.lip_color)
    return img


def apply_warp(img: np.ndarray, plan: FacePlan) -> np.ndarray:
    if not any(s.slim > 0 or s.eye_enlarge > 0 for _, s in plan):
        return img
    return build_warp(img.shape[:2], plan).apply(img)
