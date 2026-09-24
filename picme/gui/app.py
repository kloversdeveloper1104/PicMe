"""PicMe デスクトップ GUI (PySide6)。

- 左: 読み込んだ写真の一覧
- 中央: ビフォー/アフター比較ビュー (境界線をドラッグ)
- 右: プリセットと各種補正スライダー
設定はすべての写真に共通 (一括同期) で適用され、「一括書き出し」で全画像を出力できる。
"""

from __future__ import annotations

import multiprocessing
import sys
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QObject, QPointF, QRectF, QRunnable, QSize, Qt, QThread, QThreadPool, QTimer, Signal
from PySide6.QtGui import QAction, QColor, QIcon, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from ..analysis import Analysis, FaceAnalyzer
from ..batch import run_batch
from ..io import SUPPORTED_EXTS, ImageMeta, is_image, load_image, save_image
from ..pipeline import retouch
from ..settings import PRESET_LABELS, PRESETS, RetouchSettings

PREVIEW_MAX_SIDE = 1600

# (グループ, キー, ラベル, 最小, 最大)
SLIDERS: dict[str, list[tuple[str, str, str, int, int]]] = {
    "基本補正": [
        ("color", "exposure", "露出", -100, 100),
        ("color", "contrast", "コントラスト", -100, 100),
        ("color", "highlights", "ハイライト", -100, 100),
        ("color", "shadows", "シャドウ", -100, 100),
        ("color", "temperature", "色温度", -100, 100),
        ("color", "tint", "色かぶり補正", -100, 100),
        ("color", "vibrance", "自然な彩度", -100, 100),
        ("color", "saturation", "彩度", -100, 100),
        ("color", "sharpen", "シャープ", 0, 100),
    ],
    "美肌": [
        ("skin", "smooth", "なめらかさ", 0, 100),
        ("skin", "texture", "質感を残す", 0, 100),
        ("skin", "blemish", "シミ・ニキビ除去", 0, 100),
        ("skin", "even_tone", "肌色ムラ補正", 0, 100),
        ("skin", "brighten", "肌の明るさ", 0, 100),
    ],
    "顔・パーツ": [
        ("face", "slim", "小顔 (輪郭)", 0, 100),
        ("face", "eye_enlarge", "デカ目", 0, 100),
        ("face", "eye_brighten", "目の明るさ", 0, 100),
        ("face", "dark_circles", "クマ除去", 0, 100),
        ("face", "teeth_whiten", "歯のホワイトニング", 0, 100),
        ("face", "lip_color", "唇の血色", 0, 100),
    ],
    "背景": [
        ("background", "blur", "背景ぼかし", 0, 100),
    ],
}


def bgr_to_pixmap(img: np.ndarray) -> QPixmap:
    rgb = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    h, w = rgb.shape[:2]
    qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


def fit_preview(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    k = min(1.0, PREVIEW_MAX_SIDE / max(h, w))
    return cv2.resize(img, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA) if k < 1 else img


# ---------------------------------------------------------------------------
# 比較ビュー
# ---------------------------------------------------------------------------
class CompareView(QWidget):
    """左側に補正前、右側に補正後を表示し、境界線をドラッグで動かせるビュー。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.before: QPixmap | None = None
        self.after: QPixmap | None = None
        self.split = 0.5
        self.compare = True
        self.setMinimumSize(400, 300)
        self.setMouseTracking(True)

    def set_images(self, before: QPixmap | None, after: QPixmap | None) -> None:
        self.before, self.after = before, after
        self.update()

    def set_after(self, after: QPixmap) -> None:
        self.after = after
        self.update()

    def set_compare(self, on: bool) -> None:
        self.compare = on
        self.update()

    def _target_rect(self) -> QRectF:
        pm = self.after or self.before
        if pm is None:
            return QRectF()
        w, h = self.width() - 16, self.height() - 16
        k = min(w / pm.width(), h / pm.height())
        tw, th = pm.width() * k, pm.height() * k
        return QRectF((self.width() - tw) / 2, (self.height() - th) / 2, tw, th)

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(32, 32, 36))
        rect = self._target_rect()
        if rect.isEmpty():
            p.setPen(QColor(160, 160, 170))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "写真をドラッグ&ドロップ、または「開く」から読み込んでください")
            return
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        after = self.after or self.before
        p.drawPixmap(rect, after, QRectF(after.rect()))
        if self.compare and self.before is not None:
            sx = rect.left() + rect.width() * self.split
            src = QRectF(0, 0, self.before.width() * self.split, self.before.height())
            p.drawPixmap(QRectF(rect.left(), rect.top(), sx - rect.left(), rect.height()), self.before, src)
            p.setPen(QPen(QColor(255, 255, 255), 2))
            p.drawLine(QPointF(sx, rect.top()), QPointF(sx, rect.bottom()))
            p.setBrush(QColor(255, 255, 255))
            p.drawEllipse(QPointF(sx, rect.center().y()), 9, 9)
            p.setPen(QColor(255, 255, 255))
            p.drawText(QPointF(rect.left() + 10, rect.top() + 22), "補正前")
            p.drawText(QPointF(rect.right() - 60, rect.top() + 22), "補正後")

    def _update_split(self, x: float) -> None:
        rect = self._target_rect()
        if not rect.isEmpty():
            self.split = float(np.clip((x - rect.left()) / rect.width(), 0.0, 1.0))
            self.update()

    def mousePressEvent(self, e) -> None:
        if self.compare:
            self._update_split(e.position().x())

    def mouseMoveEvent(self, e) -> None:
        if self.compare and e.buttons() & Qt.MouseButton.LeftButton:
            self._update_split(e.position().x())


# ---------------------------------------------------------------------------
# バックグラウンド処理
# ---------------------------------------------------------------------------
class PreviewSignals(QObject):
    """ワーカースレッド → GUI スレッドへの通知。メインスレッドで生成して使い回す。"""

    done = Signal(int, object, object, object)  # generation, entry, result image, analysis
    failed = Signal(int, str)


class PreviewTask(QRunnable):
    def __init__(self, signals: PreviewSignals, gen: int, entry, analyzer: FaceAnalyzer, settings: RetouchSettings):
        super().__init__()
        self.signals, self.gen, self.entry, self.analyzer, self.settings = signals, gen, entry, analyzer, settings
        self.img, self.analysis = entry.preview, entry.analysis

    def run(self) -> None:
        try:
            analysis = self.analysis or self.analyzer.analyze(self.img)
            out = retouch(self.img, analysis, self.settings)
            self.signals.done.emit(self.gen, self.entry, out, analysis)
        except Exception as exc:
            self.signals.failed.emit(self.gen, f"{type(exc).__name__}: {exc}")


class BatchThread(QThread):
    progress = Signal(int, int, str)
    finished_all = Signal(object)

    def __init__(self, files, out_dir, settings, suffix):
        super().__init__()
        self.files, self.out_dir, self.settings, self.suffix = files, out_dir, settings, suffix

    def run(self) -> None:
        results = run_batch(
            self.files, self.out_dir, self.settings, suffix=self.suffix,
            progress=lambda d, t, r: self.progress.emit(d, t, r.source.name),
            mp_context=multiprocessing.get_context("spawn"),
        )
        self.finished_all.emit(results)


# ---------------------------------------------------------------------------
# メインウィンドウ
# ---------------------------------------------------------------------------
class ImageEntry:
    def __init__(self, path: Path):
        self.path = path
        self.preview: np.ndarray | None = None
        self.analysis: Analysis | None = None


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PicMe - AI 自動レタッチ")
        self.resize(1400, 860)
        self.setAcceptDrops(True)

        self.analyzer = FaceAnalyzer()
        self.settings = PRESETS["natural"].copy()
        self.entries: list[ImageEntry] = []
        self.current: ImageEntry | None = None
        self.generation = 0
        self.pool = QThreadPool.globalInstance()
        self.preview_signals = PreviewSignals(self)
        self.preview_signals.done.connect(self._preview_done)
        self.preview_signals.failed.connect(lambda _gen, msg: self.status.showMessage(f"処理エラー: {msg}", 8000))
        self.debounce = QTimer(self, singleShot=True, interval=120)
        self.debounce.timeout.connect(self._start_preview)
        self._updating_controls = False
        self.batch_thread: BatchThread | None = None

        self._build_ui()
        self._sync_controls()
        ai = "AI モデル: 有効" if self.analyzer.ai_available else "AI モデル: 無効 (色補正と簡易美肌のみ)"
        self.status.showMessage(ai)

    # ---- UI 構築 --------------------------------------------------------
    def _build_ui(self) -> None:
        tb = QToolBar("メイン")
        tb.setMovable(False)
        self.addToolBar(tb)
        for text, slot in (
            ("画像を開く", self.open_files),
            ("フォルダを開く", self.open_folder),
            ("この画像を書き出し", self.export_current),
            ("一括書き出し", self.export_all),
            ("プリセット読込", self.load_preset),
            ("プリセット保存", self.save_preset),
        ):
            act = QAction(text, self)
            act.triggered.connect(slot)
            tb.addAction(act)

        self.list = QListWidget()
        self.list.setIconSize(QSize(96, 96))
        self.list.setMinimumWidth(170)
        self.list.currentRowChanged.connect(self._select_row)

        self.view = CompareView()

        panel = QWidget()
        pv = QVBoxLayout(panel)
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("プリセット"))
        self.preset_combo = QComboBox()
        for key, label in PRESET_LABELS.items():
            self.preset_combo.addItem(label, key)
        self.preset_combo.setCurrentIndex(list(PRESET_LABELS).index("natural"))
        self.preset_combo.currentIndexChanged.connect(self._apply_preset)
        preset_row.addWidget(self.preset_combo, 1)
        pv.addLayout(preset_row)

        toggles = QHBoxLayout()
        self.chk_compare = QCheckBox("比較表示")
        self.chk_compare.setChecked(True)
        self.chk_compare.toggled.connect(self.view.set_compare)
        self.chk_wb = QCheckBox("自動WB")
        self.chk_tone = QCheckBox("自動トーン")
        self.chk_wb.toggled.connect(lambda v: self._set_value("color", "auto_white_balance", v))
        self.chk_tone.toggled.connect(lambda v: self._set_value("color", "auto_tone", v))
        for w in (self.chk_compare, self.chk_wb, self.chk_tone):
            toggles.addWidget(w)
        pv.addLayout(toggles)

        self.sliders: dict[tuple[str, str], tuple[QSlider, QLabel]] = {}
        for title, items in SLIDERS.items():
            box = QGroupBox(title)
            form = QFormLayout(box)
            for group, key, label, lo, hi in items:
                row = QHBoxLayout()
                slider = QSlider(Qt.Orientation.Horizontal)
                slider.setRange(lo, hi)
                value = QLabel("0")
                value.setFixedWidth(34)
                value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                slider.valueChanged.connect(lambda v, g=group, k=key, lab=value: (lab.setText(str(v)), self._set_value(g, k, v)))
                row.addWidget(slider, 1)
                row.addWidget(value)
                form.addRow(label, row)
                self.sliders[(group, key)] = (slider, value)
            pv.addWidget(box)

        reset = QPushButton("補正をリセット")
        reset.clicked.connect(lambda: self._load_settings(PRESETS["none"].copy()))
        pv.addWidget(reset)
        pv.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(340)

        splitter = QSplitter()
        splitter.addWidget(self.list)
        splitter.addWidget(self.view)
        splitter.addWidget(scroll)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([190, 860, 350])
        self.setCentralWidget(splitter)

        self.status = QStatusBar()
        self.setStatusBar(self.status)

    # ---- 設定 ↔ コントロール ----------------------------------------------
    def _sync_controls(self) -> None:
        self._updating_controls = True
        for (group, key), (slider, label) in self.sliders.items():
            v = int(round(getattr(getattr(self.settings, group), key)))
            slider.setValue(v)
            label.setText(str(v))
        self.chk_wb.setChecked(self.settings.color.auto_white_balance)
        self.chk_tone.setChecked(self.settings.color.auto_tone)
        self._updating_controls = False

    def _set_value(self, group: str, key: str, value) -> None:
        if self._updating_controls:
            return
        setattr(getattr(self.settings, group), key, value if isinstance(value, bool) else float(value))
        self.schedule_preview()

    def _load_settings(self, settings: RetouchSettings) -> None:
        self.settings = settings
        self._sync_controls()
        self.schedule_preview()

    def _apply_preset(self, _index: int) -> None:
        self._load_settings(PRESETS[self.preset_combo.currentData()].copy())

    # ---- 画像の読み込み --------------------------------------------------
    def add_paths(self, paths: list[Path]) -> None:
        first_new = len(self.entries)
        for p in paths:
            if p.is_dir():
                self.add_paths(sorted(f for f in p.iterdir() if f.is_file() and is_image(f)))
                continue
            if not is_image(p) or any(e.path == p for e in self.entries):
                continue
            try:
                img, _ = load_image(p)
            except Exception as exc:
                self.status.showMessage(f"読み込み失敗: {p.name} ({exc})", 5000)
                continue
            entry = ImageEntry(p)
            entry.preview = fit_preview(img)
            self.entries.append(entry)
            thumb = cv2.resize(entry.preview, None, fx=96 / max(entry.preview.shape[:2]), fy=96 / max(entry.preview.shape[:2]))
            item = QListWidgetItem(QIcon(bgr_to_pixmap(thumb)), p.name)
            self.list.addItem(item)
        if len(self.entries) > first_new and self.current is None:
            self.list.setCurrentRow(first_new)
        self.status.showMessage(f"{len(self.entries)} 枚の写真", 3000)

    def open_files(self) -> None:
        pattern = " ".join(f"*{e}" for e in sorted(SUPPORTED_EXTS))
        files, _ = QFileDialog.getOpenFileNames(self, "画像を開く", "", f"画像 ({pattern})")
        self.add_paths([Path(f) for f in files])

    def open_folder(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "フォルダを開く")
        if d:
            self.add_paths([Path(d)])

    def dragEnterEvent(self, e) -> None:
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e) -> None:
        self.add_paths([Path(u.toLocalFile()) for u in e.mimeData().urls()])

    def _select_row(self, row: int) -> None:
        if row < 0 or row >= len(self.entries):
            return
        self.current = self.entries[row]
        self.view.set_images(bgr_to_pixmap(self.current.preview), None)
        self.schedule_preview()

    # ---- プレビュー --------------------------------------------------------
    def schedule_preview(self) -> None:
        if self.current is not None:
            self.debounce.start()

    def _start_preview(self) -> None:
        entry = self.current
        if entry is None:
            return
        self.generation += 1
        task = PreviewTask(self.preview_signals, self.generation, entry, self.analyzer, self.settings.copy())
        self.status.showMessage("処理中…")
        self.pool.start(task)

    def _preview_done(self, gen: int, entry: ImageEntry, out: np.ndarray, analysis: Analysis) -> None:
        entry.analysis = analysis
        if gen != self.generation or entry is not self.current:
            return  # 古い結果は捨てる
        self.view.set_after(bgr_to_pixmap(out))
        self.status.showMessage(f"{entry.path.name}  検出した顔: {len(analysis.faces)}")

    # ---- 書き出し --------------------------------------------------------
    def export_current(self) -> None:
        if self.current is None:
            return
        src = self.current.path
        dst, _ = QFileDialog.getSaveFileName(self, "書き出し", str(src.with_name(f"{src.stem}_picme{src.suffix}")))
        if not dst:
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            img, meta = load_image(src)
            # 解析はフル解像度でやり直す (ランドマーク精度のため)
            out = retouch(img, self.analyzer.analyze(img), self.settings)
            save_image(dst, out, meta)
            self.status.showMessage(f"保存しました: {dst}", 5000)
        except Exception as exc:
            QMessageBox.critical(self, "エラー", f"書き出しに失敗しました:\n{exc}")
        finally:
            QApplication.restoreOverrideCursor()

    def export_all(self) -> None:
        if not self.entries or self.batch_thread is not None:
            return
        d = QFileDialog.getExistingDirectory(self, "書き出し先フォルダ")
        if not d:
            return
        files = [e.path for e in self.entries]
        dlg = QProgressDialog("一括書き出し中…", None, 0, len(files), self)
        dlg.setWindowModality(Qt.WindowModality.WindowModal)
        dlg.setMinimumDuration(0)
        self.batch_thread = BatchThread(files, Path(d), self.settings.copy(), "_picme")

        def on_progress(done, total, name):
            dlg.setMaximum(total)
            dlg.setValue(done)
            dlg.setLabelText(f"{name} ({done}/{total})")

        def on_done(results):
            dlg.close()
            failed = [r for r in results if r.error]
            msg = f"{len(results) - len(failed)} 枚を書き出しました。"
            if failed:
                msg += "\n失敗:\n" + "\n".join(f"{r.source.name}: {r.error}" for r in failed[:10])
            QMessageBox.information(self, "一括書き出し", msg)
            self.batch_thread = None

        self.batch_thread.progress.connect(on_progress)
        self.batch_thread.finished_all.connect(on_done)
        self.batch_thread.start()

    # ---- プリセット ファイル ---------------------------------------------
    def load_preset(self) -> None:
        f, _ = QFileDialog.getOpenFileName(self, "プリセット読込", "", "プリセット (*.json)")
        if f:
            try:
                self._load_settings(RetouchSettings.load(f))
            except Exception as exc:
                QMessageBox.critical(self, "エラー", f"読み込めませんでした:\n{exc}")

    def save_preset(self) -> None:
        f, _ = QFileDialog.getSaveFileName(self, "プリセット保存", "my_preset.json", "プリセット (*.json)")
        if f:
            self.settings.save(f)
            self.status.showMessage(f"プリセットを保存しました: {f}", 5000)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv if argv is None else argv
    app = QApplication(argv)
    app.setApplicationName("PicMe")
    win = MainWindow()
    win.add_paths([Path(a) for a in argv[1:]])
    win.show()
    return app.exec()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
