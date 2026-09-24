"""ビフォー/アフター比較ビュー (拡大縮小・パン・修正ブラシ・人物番号の表示)。"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QWidget

MODE_VIEW = "view"
MODE_BRUSH = "brush"


class CompareView(QWidget):
    """左側に補正前、右側に補正後を表示し、境界線をドラッグで動かせるビュー。

    - ホイール: 拡大縮小 (カーソル位置を中心に)
    - 右ドラッグ / 中ドラッグ / Space+ドラッグ: 表示位置の移動
    - 修正ブラシモード: クリック・ドラッグで修正箇所を追加
    - 人物番号をクリック: その人を選択
    """

    spotAdded = Signal(float, float, float)  # x, y (0..1), 半径 (長辺に対する割合)
    faceClicked = Signal(int)
    zoomChanged = Signal(float)  # 元画像に対する表示倍率

    def __init__(self, parent=None):
        super().__init__(parent)
        self.before: QPixmap | None = None
        self.after: QPixmap | None = None
        self.split = 0.5
        self.compare = True
        self.mode = MODE_VIEW
        self.brush_size = 0.01  # 長辺に対する割合
        self.source_scale = 1.0  # 表示中の画像 1px が元画像の何 px か
        self.faces: list[tuple[float, float, str, bool]] = []  # x, y (0..1), ラベル, 選択中
        self.show_faces = True
        self._zoom: float | None = None  # None = 全体表示
        self._center = QPointF(0, 0)  # 表示中心 (画像座標)
        self._drag: str | None = None
        self._last_pos = QPointF()
        self._last_spot: QPointF | None = None
        self._mouse = QPointF(-1, -1)
        self._space = False
        self.setMinimumSize(400, 300)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    # ---- 画像の設定 ------------------------------------------------------
    def set_images(self, before: QPixmap | None, after: QPixmap | None, reset_view: bool = True) -> None:
        old = self.before
        self.before, self.after = before, after
        if reset_view or old is None or before is None or old.size() != before.size():
            if old is not None and before is not None and not reset_view and self._zoom is not None:
                # 解像度だけ変わった (プレビュー ↔ 等倍) 場合は見ている場所を保つ
                k = before.width() / old.width()
                self._zoom /= k
                self._center = QPointF(self._center.x() * k, self._center.y() * k)
            else:
                self._zoom = None
                if before is not None:
                    self._center = QPointF(before.width() / 2, before.height() / 2)
        self._emit_zoom()
        self.update()

    def set_after(self, after: QPixmap) -> None:
        self.after = after
        self.update()

    def set_compare(self, on: bool) -> None:
        self.compare = on
        self.update()

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self.setCursor(Qt.CursorShape.BlankCursor if mode == MODE_BRUSH else Qt.CursorShape.ArrowCursor)
        self.update()

    def set_faces(self, faces: list[tuple[float, float, str, bool]]) -> None:
        self.faces = faces
        self.update()

    # ---- 座標変換 --------------------------------------------------------
    def _pixmap(self) -> QPixmap | None:
        return self.after or self.before

    def _fit_scale(self) -> float:
        pm = self._pixmap()
        if pm is None:
            return 1.0
        return min((self.width() - 16) / pm.width(), (self.height() - 16) / pm.height())

    def scale(self) -> float:
        return self._fit_scale() if self._zoom is None else self._zoom

    def _image_rect(self) -> QRectF:
        pm = self._pixmap()
        if pm is None:
            return QRectF()
        s = self.scale()
        if self._zoom is None:
            cx, cy = pm.width() / 2, pm.height() / 2
        else:
            cx, cy = self._center.x(), self._center.y()
        left = self.width() / 2 - cx * s
        top = self.height() / 2 - cy * s
        return QRectF(left, top, pm.width() * s, pm.height() * s)

    def to_image(self, p: QPointF) -> QPointF:
        r = self._image_rect()
        s = self.scale()
        return QPointF((p.x() - r.left()) / s, (p.y() - r.top()) / s)

    # ---- 拡大縮小 --------------------------------------------------------
    def fit(self) -> None:
        self._zoom = None
        self._emit_zoom()
        self.update()

    def zoom_to(self, zoom: float, anchor: QPointF | None = None) -> None:
        pm = self._pixmap()
        if pm is None:
            return
        anchor = anchor or QPointF(self.width() / 2, self.height() / 2)
        img_pt = self.to_image(anchor)
        zoom = float(np.clip(zoom, self._fit_scale() * 0.5, 32.0))
        self._zoom = zoom
        # anchor の下の画素が動かないように表示中心を調整
        self._center = QPointF(
            img_pt.x() - (anchor.x() - self.width() / 2) / zoom,
            img_pt.y() - (anchor.y() - self.height() / 2) / zoom,
        )
        self._emit_zoom()
        self.update()

    def actual_size(self) -> None:
        """元画像の 1px = 画面の 1px (等倍) で表示。"""
        self.zoom_to(self.source_scale)

    def _emit_zoom(self) -> None:
        self.zoomChanged.emit(self.scale() / max(self.source_scale, 1e-6))

    def wheelEvent(self, e) -> None:
        if self._pixmap() is None:
            return
        factor = 1.25 if e.angleDelta().y() > 0 else 0.8
        self.zoom_to(self.scale() * factor, e.position())

    def resizeEvent(self, e) -> None:
        self._emit_zoom()
        super().resizeEvent(e)

    # ---- 描画 ------------------------------------------------------------
    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(32, 32, 36))
        pm = self._pixmap()
        if pm is None:
            p.setPen(QColor(160, 160, 170))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "写真をドラッグ&ドロップ、または「画像を開く」から読み込んでください")
            return
        rect = self._image_rect()
        s = self.scale()
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, s < 2.0)
        p.drawPixmap(rect, pm, QRectF(pm.rect()))
        if self.compare and self.before is not None and self.after is not None:
            sx = rect.left() + rect.width() * self.split
            src = QRectF(0, 0, self.before.width() * self.split, self.before.height())
            p.drawPixmap(QRectF(rect.left(), rect.top(), sx - rect.left(), rect.height()), self.before, src)
            p.setPen(QPen(QColor(255, 255, 255), 2))
            p.drawLine(QPointF(sx, max(0, rect.top())), QPointF(sx, min(self.height(), rect.bottom())))
            p.setBrush(QColor(255, 255, 255))
            p.drawEllipse(QPointF(sx, self.height() / 2), 9, 9)
            p.setPen(QColor(255, 255, 255))
            p.drawText(QPointF(12, 22), "補正前")
            p.drawText(QPointF(self.width() - 60, 22), "補正後")

        if self.show_faces and self.faces:
            font = QFont(p.font())
            font.setBold(True)
            p.setFont(font)
            for x, y, label, selected in self.faces:
                c = QPointF(rect.left() + x * rect.width(), rect.top() + y * rect.height())
                p.setPen(QPen(QColor(255, 255, 255), 2))
                p.setBrush(QColor(255, 140, 0, 230) if selected else QColor(40, 120, 255, 200))
                p.drawEllipse(c, 12, 12)
                p.drawText(QRectF(c.x() - 12, c.y() - 12, 24, 24), Qt.AlignmentFlag.AlignCenter, label)

        if self.mode == MODE_BRUSH and self._mouse.x() >= 0:
            r = self.brush_size * max(pm.width(), pm.height()) * s
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(QColor(0, 0, 0), 3))
            p.drawEllipse(self._mouse, r, r)
            p.setPen(QPen(QColor(255, 255, 255), 1.5))
            p.drawEllipse(self._mouse, r, r)

    # ---- マウス ----------------------------------------------------------
    def _face_at(self, pos: QPointF) -> int | None:
        rect = self._image_rect()
        for i, (x, y, _, _) in enumerate(self.faces):
            c = QPointF(rect.left() + x * rect.width(), rect.top() + y * rect.height())
            if (c - pos).manhattanLength() < 20:
                return i
        return None

    def _add_spot(self, pos: QPointF) -> None:
        pm = self._pixmap()
        img_pt = self.to_image(pos)
        if not (0 <= img_pt.x() < pm.width() and 0 <= img_pt.y() < pm.height()):
            return
        # ドラッグ中は半径の 0.8 倍ごとに点を打つ
        side = max(pm.width(), pm.height())
        if self._last_spot is not None:
            if (img_pt - self._last_spot).manhattanLength() < self.brush_size * side * 0.8:
                return
        self._last_spot = img_pt
        self.spotAdded.emit(img_pt.x() / pm.width(), img_pt.y() / pm.height(), self.brush_size)

    def _update_split(self, x: float) -> None:
        rect = self._image_rect()
        if not rect.isEmpty():
            self.split = float(np.clip((x - rect.left()) / rect.width(), 0.0, 1.0))
            self.update()

    def mousePressEvent(self, e) -> None:
        if self._pixmap() is None:
            return
        pos = e.position()
        self._last_pos = pos
        if e.button() in (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton) or self._space:
            self._drag = "pan"
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        if e.button() != Qt.MouseButton.LeftButton:
            return
        if self.mode == MODE_BRUSH:
            self._drag = "brush"
            self._last_spot = None
            self._add_spot(pos)
            return
        face = self._face_at(pos) if self.show_faces else None
        if face is not None:
            self.faceClicked.emit(face)
            return
        if self.compare:
            self._drag = "split"
            self._update_split(pos.x())
        else:
            self._drag = "pan"

    def mouseMoveEvent(self, e) -> None:
        pos = e.position()
        self._mouse = pos
        if self._drag == "pan":
            if self._zoom is None:
                self._zoom = self._fit_scale()
            d = pos - self._last_pos
            self._center -= QPointF(d.x() / self._zoom, d.y() / self._zoom)
        elif self._drag == "split":
            self._update_split(pos.x())
        elif self._drag == "brush":
            self._add_spot(pos)
        self._last_pos = pos
        self.update()

    def mouseReleaseEvent(self, e) -> None:
        self._drag = None
        self._last_spot = None
        self.set_mode(self.mode)

    def leaveEvent(self, e) -> None:
        self._mouse = QPointF(-1, -1)
        self.update()

    def keyPressEvent(self, e) -> None:
        if e.key() == Qt.Key.Key_Space and not e.isAutoRepeat():
            self._space = True
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        else:
            super().keyPressEvent(e)

    def keyReleaseEvent(self, e) -> None:
        if e.key() == Qt.Key.Key_Space and not e.isAutoRepeat():
            self._space = False
            self.set_mode(self.mode)
        else:
            super().keyReleaseEvent(e)
