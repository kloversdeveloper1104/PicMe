"""PicMe デスクトップ GUI (PySide6)。

- 左: 読み込んだ写真の一覧 (★ = 個別設定あり、✎ = 修正ブラシ / 人物ごとの設定あり)
- 中央: ビフォー/アフター比較ビュー (拡大縮小・パン・修正ブラシ)
- 右: プリセットと各種補正スライダー

基本の設定はすべての写真に共通で適用され (一括同期)、必要な写真だけ
「この写真だけ個別に調整」で上書きできる。
"""

from __future__ import annotations

import copy
import multiprocessing
import sys
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QObject, QRunnable, QSize, Qt, QThread, QThreadPool, QTimer, Signal
from PySide6.QtGui import QAction, QIcon, QImage, QKeySequence, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QColorDialog,
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
    QRadioButton,
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
from ..io import READ_ONLY_EXTS, SUPPORTED_EXTS, is_image, load_image, save_image
from ..pipeline import match_override, retouch
from ..settings import PRESET_LABELS, PRESETS, FaceOverride, RetouchSettings
from .history import History
from .view import MODE_BRUSH, MODE_VIEW, CompareView

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
        ("skin", "wrinkles", "しわ・ほうれい線", 0, 100),
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
        ("face", "hair_shine", "髪のツヤ", 0, 100),
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
# データ
# ---------------------------------------------------------------------------
class ImageEntry:
    def __init__(self, path: Path, preview: np.ndarray, orig_shape: tuple[int, int]):
        self.path = path
        self.preview = preview
        self.orig_shape = orig_shape
        self.full: np.ndarray | None = None  # 等倍プレビュー用 (必要になったら読み込む)
        self.analysis: Analysis | None = None
        self.own: RetouchSettings | None = None  # 個別設定 (None なら共通設定)
        self.spots: list[list[float]] = []
        self.overrides: list[FaceOverride] = []

    def state(self) -> dict:
        return {
            "own": self.own.to_dict(include_image_data=False) if self.own else None,
            "spots": copy.deepcopy(self.spots),
            "overrides": [o.to_dict() for o in self.overrides],
        }

    def restore(self, st: dict) -> None:
        self.own = RetouchSettings.from_dict(st["own"]) if st.get("own") else None
        self.spots = copy.deepcopy(st.get("spots", []))
        self.overrides = [FaceOverride.from_dict(o) for o in st.get("overrides", [])]

    @property
    def has_edits(self) -> bool:
        return bool(self.spots or self.overrides)


# ---------------------------------------------------------------------------
# バックグラウンド処理
# ---------------------------------------------------------------------------
class PreviewSignals(QObject):
    """ワーカースレッド → GUI スレッドへの通知。メインスレッドで生成して使い回す。"""

    done = Signal(int, object, object, object)  # generation, entry, result image, analysis
    failed = Signal(int, str)


class PreviewTask(QRunnable):
    def __init__(self, signals: PreviewSignals, gen: int, entry: ImageEntry, work: np.ndarray, analyzer: FaceAnalyzer, settings: RetouchSettings):
        super().__init__()
        self.signals, self.gen, self.entry, self.work = signals, gen, entry, work
        self.analyzer, self.settings = analyzer, settings
        self.analysis = entry.analysis

    def run(self) -> None:
        try:
            analysis = self.analysis or self.analyzer.analyze(self.entry.preview)
            out = retouch(self.work, analysis, self.settings)
            self.signals.done.emit(self.gen, self.entry, out, analysis)
        except Exception as exc:
            self.signals.failed.emit(self.gen, f"{type(exc).__name__}: {exc}")


class BatchThread(QThread):
    progress = Signal(int, int, str)
    finished_all = Signal(object)

    def __init__(self, files, out_dir, settings, per_file, suffix):
        super().__init__()
        self.files, self.out_dir, self.settings, self.per_file, self.suffix = files, out_dir, settings, per_file, suffix

    def run(self) -> None:
        results = run_batch(
            self.files, self.out_dir, self.settings, suffix=self.suffix, per_file=self.per_file,
            progress=lambda d, t, r: self.progress.emit(d, t, r.source.name),
            mp_context=multiprocessing.get_context("spawn"),
        )
        self.finished_all.emit(results)


# ---------------------------------------------------------------------------
# メインウィンドウ
# ---------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self, analyzer: FaceAnalyzer | None = None):
        super().__init__()
        self.setWindowTitle("PicMe - AI 自動レタッチ")
        self.resize(1440, 900)
        self.setAcceptDrops(True)

        self.analyzer = analyzer or FaceAnalyzer()
        self.global_settings = PRESETS["natural"].copy()
        self.entries: list[ImageEntry] = []
        self.current: ImageEntry | None = None
        self.person = 0  # 0 = 全員、1.. = 人物番号
        self.generation = 0
        self.pool = QThreadPool.globalInstance()
        self.preview_signals = PreviewSignals(self)
        self.preview_signals.done.connect(self._preview_done)
        self.preview_signals.failed.connect(lambda _gen, msg: self.status.showMessage(f"処理エラー: {msg}", 8000))
        self.debounce = QTimer(self, singleShot=True, interval=120)
        self.debounce.timeout.connect(self._start_preview)
        self.history = History()
        self.history_timer = QTimer(self, singleShot=True, interval=400)
        self.history_timer.timeout.connect(self._commit_history)
        self._updating = False
        self.batch_thread: BatchThread | None = None

        self._build_ui()
        self._sync_controls()
        self.history.reset(self.snapshot())
        ai = "AI モデル: 有効" if self.analyzer.ai_available else "AI モデル: 無効 (色補正と簡易美肌のみ)"
        self.status.showMessage(ai)

    # ======================================================================
    # UI 構築
    # ======================================================================
    def _action(self, text: str, slot, shortcut=None, tip: str = "") -> QAction:
        act = QAction(text, self)
        act.triggered.connect(slot)
        if shortcut:
            act.setShortcuts(shortcut if isinstance(shortcut, list) else [shortcut])
        if tip:
            act.setToolTip(tip)
        return act

    def _build_ui(self) -> None:
        tb = QToolBar("メイン")
        tb.setMovable(False)
        self.addToolBar(tb)
        tb.addAction(self._action("画像を開く", self.open_files, QKeySequence.StandardKey.Open))
        tb.addAction(self._action("フォルダを開く", self.open_folder))
        tb.addSeparator()
        self.act_undo = self._action("元に戻す", self.undo, QKeySequence.StandardKey.Undo, "Ctrl+Z")
        self.act_redo = self._action("やり直し", self.redo, [QKeySequence("Ctrl+Shift+Z"), QKeySequence("Ctrl+Y")], "Ctrl+Shift+Z / Ctrl+Y")
        tb.addAction(self.act_undo)
        tb.addAction(self.act_redo)
        tb.addSeparator()
        tb.addAction(self._action("全体表示", lambda: self.view.fit(), QKeySequence("Ctrl+0")))
        tb.addAction(self._action("等倍 (100%)", self.actual_size, QKeySequence("Ctrl+1")))
        self.chk_hq = QCheckBox("高画質プレビュー")
        self.chk_hq.setToolTip("元の解像度でプレビューします (拡大して肌の質感を確認するとき用。処理は遅くなります)")
        self.chk_hq.toggled.connect(self._toggle_hq)
        tb.addWidget(self.chk_hq)
        tb.addSeparator()
        tb.addAction(self._action("この画像を書き出し", self.export_current, QKeySequence("Ctrl+E")))
        tb.addAction(self._action("一括書き出し", self.export_all, QKeySequence("Ctrl+Shift+E")))
        tb.addSeparator()
        tb.addAction(self._action("プリセット読込", self.load_preset))
        tb.addAction(self._action("プリセット保存", self.save_preset))

        self.list = QListWidget()
        self.list.setIconSize(QSize(96, 96))
        self.list.setMinimumWidth(170)
        self.list.currentRowChanged.connect(self._select_row)

        self.view = CompareView()
        self.view.spotAdded.connect(self._add_spot)
        self.view.faceClicked.connect(lambda i: self.person_combo.setCurrentIndex(i + 1))
        self.view.zoomChanged.connect(self._zoom_changed)

        # ---- 右パネル ----
        panel = QWidget()
        pv = QVBoxLayout(panel)

        row = QHBoxLayout()
        row.addWidget(QLabel("プリセット"))
        self.preset_combo = QComboBox()
        self.preset_combo.addItem("― 選択して適用 ―", None)
        for key, label in PRESET_LABELS.items():
            self.preset_combo.addItem(label, key)
        self.preset_combo.activated.connect(self._apply_preset)
        row.addWidget(self.preset_combo, 1)
        pv.addLayout(row)

        self.chk_own = QCheckBox("この写真だけ個別に調整")
        self.chk_own.setToolTip("オフのときはすべての写真に共通の設定を編集します")
        self.chk_own.toggled.connect(self._toggle_own)
        pv.addWidget(self.chk_own)
        self.target_label = QLabel()
        self.target_label.setStyleSheet("color: gray;")
        pv.addWidget(self.target_label)

        toggles = QHBoxLayout()
        self.chk_compare = QCheckBox("比較表示")
        self.chk_compare.setChecked(True)
        self.chk_compare.toggled.connect(self.view.set_compare)
        self.chk_wb = QCheckBox("自動WB")
        self.chk_tone = QCheckBox("自動トーン")
        self.chk_wb.toggled.connect(lambda v: self._on_value("color", "auto_white_balance", v))
        self.chk_tone.toggled.connect(lambda v: self._on_value("color", "auto_tone", v))
        for w in (self.chk_compare, self.chk_wb, self.chk_tone):
            toggles.addWidget(w)
        pv.addLayout(toggles)

        # ツール (修正ブラシ)
        tools = QGroupBox("ツール")
        tl = QVBoxLayout(tools)
        trow = QHBoxLayout()
        self.rb_view = QRadioButton("表示 (V)")
        self.rb_brush = QRadioButton("修正ブラシ (B)")
        self.rb_view.setChecked(True)
        grp = QButtonGroup(self)
        grp.addButton(self.rb_view)
        grp.addButton(self.rb_brush)
        self.rb_brush.toggled.connect(lambda on: self.view.set_mode(MODE_BRUSH if on else MODE_VIEW))
        trow.addWidget(self.rb_view)
        trow.addWidget(self.rb_brush)
        tl.addLayout(trow)
        brow = QHBoxLayout()
        brow.addWidget(QLabel("ブラシの大きさ"))
        self.brush_slider = QSlider(Qt.Orientation.Horizontal)
        self.brush_slider.setRange(2, 60)
        self.brush_slider.setValue(12)
        self.brush_slider.valueChanged.connect(lambda v: setattr(self.view, "brush_size", v / 1000.0))
        self.view.brush_size = 0.012
        brow.addWidget(self.brush_slider, 1)
        tl.addLayout(brow)
        crow = QHBoxLayout()
        self.spot_label = QLabel("修正箇所: 0")
        clear = QPushButton("修正をすべて消去")
        clear.clicked.connect(self._clear_spots)
        crow.addWidget(self.spot_label, 1)
        crow.addWidget(clear)
        tl.addLayout(crow)
        pv.addWidget(tools)
        for key, rb in (("V", self.rb_view), ("B", self.rb_brush)):
            act = QAction(self)
            act.setShortcut(QKeySequence(key))
            act.triggered.connect(lambda _=False, r=rb: r.setChecked(True))
            self.addAction(act)

        self.sliders: dict[tuple[str, str], tuple[QSlider, QLabel]] = {}
        for title, items in SLIDERS.items():
            box = QGroupBox(title)
            form = QFormLayout(box)
            if title == "顔・パーツ":
                prow = QHBoxLayout()
                self.person_combo = QComboBox()
                self.person_combo.addItem("全員")
                self.person_combo.currentIndexChanged.connect(self._select_person)
                self.person_clear = QPushButton("解除")
                self.person_clear.setToolTip("この人だけの設定を削除して、全員共通の設定に戻す")
                self.person_clear.clicked.connect(self._clear_person)
                prow.addWidget(self.person_combo, 1)
                prow.addWidget(self.person_clear)
                form.addRow("対象", prow)
                self.chk_faces = QCheckBox("人物番号を表示")
                self.chk_faces.setChecked(True)
                self.chk_faces.toggled.connect(self._toggle_face_labels)
                form.addRow(self.chk_faces)
            for group, key, label, lo, hi in items:
                form.addRow(label, self._make_slider(group, key, lo, hi))
            if title == "顔・パーツ":
                self.person_skin_label = QLabel("この人の美肌 (%)")
                self.person_skin_row = self._make_slider("person", "skin", 0, 100)
                form.addRow(self.person_skin_label, self.person_skin_row)
            if title == "背景":
                self.bg_label = QLabel("置き換え: なし")
                self.bg_label.setWordWrap(True)
                form.addRow(self.bg_label)
                bgrow = QHBoxLayout()
                for text, slot in (("色で置き換え", self._bg_color), ("画像で置き換え", self._bg_image), ("元に戻す", self._bg_clear)):
                    b = QPushButton(text)
                    b.clicked.connect(slot)
                    bgrow.addWidget(b)
                form.addRow(bgrow)
            pv.addWidget(box)

        reset = QPushButton("補正をリセット")
        reset.clicked.connect(self._reset_settings)
        pv.addWidget(reset)
        pv.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(360)

        splitter = QSplitter()
        splitter.addWidget(self.list)
        splitter.addWidget(self.view)
        splitter.addWidget(scroll)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([190, 880, 370])
        self.setCentralWidget(splitter)

        self.status = QStatusBar()
        self.zoom_label = QLabel("")
        self.status.addPermanentWidget(self.zoom_label)
        self.setStatusBar(self.status)
        self._update_person_controls()

    def _make_slider(self, group: str, key: str, lo: int, hi: int) -> QWidget:
        w = QWidget()
        row = QHBoxLayout(w)
        row.setContentsMargins(0, 0, 0, 0)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(lo, hi)
        value = QLabel("0")
        value.setFixedWidth(34)
        value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        slider.valueChanged.connect(lambda v, g=group, k=key, lab=value: (lab.setText(str(v)), self._on_value(g, k, v)))
        row.addWidget(slider, 1)
        row.addWidget(value)
        self.sliders[(group, key)] = (slider, value)
        return w

    # ======================================================================
    # 設定の対象
    # ======================================================================
    def target_settings(self) -> RetouchSettings:
        """いま編集している設定 (個別設定があればそれ、なければ共通設定)。"""
        e = self.current
        return e.own if e is not None and e.own is not None else self.global_settings

    def effective(self, entry: ImageEntry) -> RetouchSettings:
        """その写真に実際に適用する設定。"""
        s = (entry.own or self.global_settings).copy()
        s.face_overrides = copy.deepcopy(entry.overrides)
        s.heal_spots = copy.deepcopy(entry.spots)
        return s

    def _person_override(self, create: bool) -> FaceOverride | None:
        e = self.current
        if e is None or e.analysis is None or self.person <= 0 or self.person > len(e.analysis.faces):
            return None
        f = e.analysis.faces[self.person - 1]
        o = match_override(f, e.overrides, e.analysis.shape)
        if o is None and create:
            h, w = e.analysis.shape
            c = f.center
            o = FaceOverride(x=float(c[0] / w), y=float(c[1] / h), face=copy.deepcopy(self.target_settings().face))
            e.overrides.append(o)
        return o

    def _on_value(self, group: str, key: str, value) -> None:
        if self._updating:
            return
        if group == "person":
            o = self._person_override(create=True)
            if o is None:
                return
            o.skin = float(value)
            self._update_person_controls()
        elif group == "face" and self.person > 0:
            o = self._person_override(create=True)
            if o is None:
                return
            setattr(o.face, key, float(value))
            self._update_person_controls()
        else:
            section = getattr(self.target_settings(), group)
            setattr(section, key, value if isinstance(value, bool) else float(value))
        self._changed()

    def _changed(self) -> None:
        """設定が変わったとき: プレビュー更新と履歴への記録を予約する。"""
        self.schedule_preview()
        self.history_timer.start()
        if self.current is not None:
            self._update_item(self.current)

    # ======================================================================
    # 設定 ↔ コントロール
    # ======================================================================
    def _sync_controls(self) -> None:
        self._updating = True
        s = self.target_settings()
        o = self._person_override(create=False)
        face = o.face if o is not None else s.face
        for (group, key), (slider, label) in self.sliders.items():
            if group == "person":
                v = int(round(o.skin if o is not None else 100))
            elif group == "face":
                v = int(round(getattr(face, key)))
            else:
                v = int(round(getattr(getattr(s, group), key)))
            slider.setValue(v)
            label.setText(str(v))
        self.chk_wb.setChecked(s.color.auto_white_balance)
        self.chk_tone.setChecked(s.color.auto_tone)
        e = self.current
        self.chk_own.setEnabled(e is not None)
        self.chk_own.setChecked(e is not None and e.own is not None)
        self.target_label.setText(
            "編集中: この写真だけの設定" if e is not None and e.own is not None else "編集中: 全写真に共通の設定"
        )
        bg = s.background
        if bg.replace_image:
            self.bg_label.setText(f"置き換え: 画像 ({Path(bg.replace_image).name})")
        elif bg.replace_color:
            self.bg_label.setText(f"置き換え: 色 {bg.replace_color}")
        else:
            self.bg_label.setText("置き換え: なし")
        self.spot_label.setText(f"修正箇所: {len(e.spots) if e else 0}")
        self._updating = False
        self._update_person_controls()

    def _update_person_controls(self) -> None:
        person = self.person > 0
        has_override = self._person_override(create=False) is not None
        self.person_skin_label.setVisible(person)
        self.person_skin_row.setVisible(person)
        self.person_clear.setEnabled(person and has_override)
        self._update_face_labels()

    def _update_face_labels(self) -> None:
        e = self.current
        if e is None or e.analysis is None or not self.chk_faces.isChecked():
            self.view.set_faces([])
            return
        h, w = e.analysis.shape
        self.view.set_faces(
            [(float(f.center[0] / w), float(f.center[1] / h), str(i + 1), self.person == i + 1) for i, f in enumerate(e.analysis.faces)]
        )

    def _toggle_face_labels(self, _on: bool) -> None:
        self._update_face_labels()

    def _refresh_person_combo(self) -> None:
        e = self.current
        n = len(e.analysis.faces) if e is not None and e.analysis is not None else 0
        if self.person_combo.count() == n + 1:
            return
        self._updating = True
        self.person_combo.clear()
        self.person_combo.addItem("全員")
        for i in range(n):
            self.person_combo.addItem(f"人物 {i + 1}")
        self.person = 0
        self.person_combo.setCurrentIndex(0)
        self._updating = False
        self._sync_controls()

    def _select_person(self, index: int) -> None:
        if self._updating:
            return
        self.person = max(0, index)
        self._sync_controls()

    def _clear_person(self) -> None:
        o = self._person_override(create=False)
        if o is not None and self.current is not None:
            self.current.overrides.remove(o)
            self._sync_controls()
            self._changed()

    def _apply_preset(self, index: int) -> None:
        key = self.preset_combo.itemData(index)
        self.preset_combo.setCurrentIndex(0)
        if key is None:
            return
        self._replace_target(PRESETS[key].copy())

    def _replace_target(self, settings: RetouchSettings) -> None:
        if self.current is not None and self.current.own is not None:
            self.current.own = settings
        else:
            self.global_settings = settings
        self._sync_controls()
        self._changed()

    def _reset_settings(self) -> None:
        self._replace_target(PRESETS["none"].copy())

    def _toggle_own(self, on: bool) -> None:
        if self._updating or self.current is None:
            return
        self.current.own = self.global_settings.preset_only() if on else None
        self._sync_controls()
        self._changed()

    # ---- 修正ブラシ ------------------------------------------------------
    def _add_spot(self, x: float, y: float, r: float) -> None:
        if self.current is None:
            return
        self.current.spots.append([x, y, r])
        self.spot_label.setText(f"修正箇所: {len(self.current.spots)}")
        self._changed()

    def _clear_spots(self) -> None:
        if self.current is not None and self.current.spots:
            self.current.spots.clear()
            self._sync_controls()
            self._changed()

    # ---- 背景 ------------------------------------------------------------
    def _bg_color(self) -> None:
        color = QColorDialog.getColor(parent=self, title="背景色")
        if color.isValid():
            bg = self.target_settings().background
            bg.replace_color, bg.replace_image = color.name(), ""
            self._sync_controls()
            self._changed()

    def _bg_image(self) -> None:
        pattern = " ".join(f"*{e}" for e in sorted(SUPPORTED_EXTS))
        f, _ = QFileDialog.getOpenFileName(self, "背景画像", "", f"画像 ({pattern})")
        if f:
            bg = self.target_settings().background
            bg.replace_image, bg.replace_color = f, ""
            self._sync_controls()
            self._changed()

    def _bg_clear(self) -> None:
        bg = self.target_settings().background
        bg.replace_image = bg.replace_color = ""
        self._sync_controls()
        self._changed()

    # ======================================================================
    # 元に戻す / やり直し
    # ======================================================================
    def snapshot(self) -> dict:
        return {
            "global": self.global_settings.to_dict(include_image_data=False),
            "entries": {str(e.path): e.state() for e in self.entries},
        }

    def restore(self, state: dict) -> None:
        self.global_settings = RetouchSettings.from_dict(state["global"])
        for e in self.entries:
            st = state["entries"].get(str(e.path))
            if st is not None:
                e.restore(st)
            self._update_item(e)
        self._sync_controls()
        self.schedule_preview()

    def _commit_history(self) -> None:
        self.history.commit(self.snapshot())
        self._update_undo_actions()

    def _update_undo_actions(self) -> None:
        self.act_undo.setEnabled(self.history.can_undo)
        self.act_redo.setEnabled(self.history.can_redo)

    def undo(self) -> None:
        self._flush_history()
        if (state := self.history.undo()) is not None:
            self.restore(state)
        self._update_undo_actions()

    def redo(self) -> None:
        self._flush_history()
        if (state := self.history.redo()) is not None:
            self.restore(state)
        self._update_undo_actions()

    def _flush_history(self) -> None:
        if self.history_timer.isActive():
            self.history_timer.stop()
            self._commit_history()

    # ======================================================================
    # 画像の読み込み
    # ======================================================================
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
            entry = ImageEntry(p, fit_preview(img), img.shape[:2])
            self.entries.append(entry)
            k = 96 / max(entry.preview.shape[:2])
            thumb = cv2.resize(entry.preview, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
            self.list.addItem(QListWidgetItem(QIcon(bgr_to_pixmap(thumb)), p.name))
        if len(self.entries) > first_new:
            self.history.reset(self.snapshot())
            self._update_undo_actions()
            if self.current is None:
                self.list.setCurrentRow(first_new)
        self.status.showMessage(f"{len(self.entries)} 枚の写真", 3000)

    def _update_item(self, entry: ImageEntry) -> None:
        row = self.entries.index(entry)
        marks = (" ★" if entry.own is not None else "") + (" ✎" if entry.has_edits else "")
        self.list.item(row).setText(entry.path.name + marks)

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
        self.person = 0
        self._show_current(reset_view=True)
        self._refresh_person_combo()
        self._sync_controls()
        self.schedule_preview()

    # ======================================================================
    # プレビュー
    # ======================================================================
    def _work_image(self, entry: ImageEntry) -> np.ndarray:
        if self.chk_hq.isChecked():
            if entry.full is None:
                QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
                try:
                    entry.full = load_image(entry.path)[0]
                finally:
                    QApplication.restoreOverrideCursor()
            return entry.full
        return entry.preview

    def _show_current(self, reset_view: bool) -> None:
        e = self.current
        if e is None:
            return
        work = self._work_image(e)
        self.view.source_scale = e.orig_shape[1] / work.shape[1]
        self.view.set_images(bgr_to_pixmap(work), None, reset_view=reset_view)

    def _toggle_hq(self, _on: bool) -> None:
        if self.current is not None:
            self._show_current(reset_view=False)
            self.schedule_preview()

    def actual_size(self) -> None:
        if self.current is None:
            return
        if not self.chk_hq.isChecked():
            self.chk_hq.setChecked(True)
        self.view.actual_size()

    def _zoom_changed(self, percent: float) -> None:
        self.zoom_label.setText(f"表示 {percent * 100:.0f}%")

    def schedule_preview(self) -> None:
        if self.current is not None:
            self.debounce.start()

    def _start_preview(self) -> None:
        entry = self.current
        if entry is None:
            return
        self.generation += 1
        task = PreviewTask(self.preview_signals, self.generation, entry, self._work_image(entry), self.analyzer, self.effective(entry))
        self.status.showMessage("処理中…")
        self.pool.start(task)

    def _preview_done(self, gen: int, entry: ImageEntry, out: np.ndarray, analysis: Analysis) -> None:
        first = entry.analysis is None
        entry.analysis = analysis
        if gen != self.generation or entry is not self.current:
            return  # 古い結果は捨てる
        if out.shape[:2] != (self.view.before.height(), self.view.before.width()):
            return  # 高画質プレビューの切り替え前の結果
        self.view.set_after(bgr_to_pixmap(out))
        if first:
            self._refresh_person_combo()
            self._update_person_controls()
        self.status.showMessage(f"{entry.path.name}  検出した顔: {len(analysis.faces)}")

    # ======================================================================
    # 書き出し
    # ======================================================================
    def export_current(self) -> None:
        if self.current is None:
            return
        src = self.current.path
        ext = ".jpg" if src.suffix.lower() in READ_ONLY_EXTS else src.suffix
        dst, _ = QFileDialog.getSaveFileName(self, "書き出し", str(src.with_name(f"{src.stem}_picme{ext}")))
        if not dst:
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            img, meta = load_image(src)
            # 解析はフル解像度でやり直す (ランドマーク精度のため)
            out = retouch(img, self.analyzer.analyze(img), self.effective(self.current))
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
        per_file = {e.path: self.effective(e) for e in self.entries if e.own is not None or e.has_edits}
        dlg = QProgressDialog("一括書き出し中…", None, 0, len(files), self)
        dlg.setWindowModality(Qt.WindowModality.WindowModal)
        dlg.setMinimumDuration(0)
        self.batch_thread = BatchThread(files, Path(d), self.global_settings.copy(), per_file, "_picme")

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

    # ======================================================================
    # プリセット ファイル
    # ======================================================================
    def load_preset(self) -> None:
        f, _ = QFileDialog.getOpenFileName(self, "プリセット読込", "", "プリセット (*.json)")
        if f:
            try:
                self._replace_target(RetouchSettings.load(f))
            except Exception as exc:
                QMessageBox.critical(self, "エラー", f"読み込めませんでした:\n{exc}")

    def save_preset(self) -> None:
        f, _ = QFileDialog.getSaveFileName(self, "プリセット保存", "my_preset.json", "プリセット (*.json)")
        if f:
            self.target_settings().save(f)
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
