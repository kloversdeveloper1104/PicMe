"""GUI の基本操作テスト (画面なしのオフスクリーンで実行)。"""

import os

import numpy as np
import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEventLoop, QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from picme.analysis import Analysis, Face, FaceAnalyzer  # noqa: E402


class FakeAnalyzer(FaceAnalyzer):
    """モデル無しで、固定位置に顔が 2 つある解析結果を返す。"""

    def __init__(self):
        super().__init__(use_ai=False)

    def analyze(self, img):
        h, w = img.shape[:2]
        faces = []
        for cx in (w * 0.3, w * 0.7):
            t = np.linspace(0, 2 * np.pi, 478, endpoint=False)
            faces.append(Face(np.stack([cx + np.cos(t) * w * 0.1, h / 2 + np.sin(t) * w * 0.1], 1).astype(np.float32)))
        return Analysis(shape=(h, w), faces=faces, skin_mask=np.full((h, w), 0.5, np.float32))


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _wait(win, ms=3000):
    loop = QEventLoop()
    for _ in range(ms // 50):
        QTimer.singleShot(50, loop.quit)
        loop.exec()
        if win.pool.activeThreadCount() == 0 and not win.debounce.isActive():
            QTimer.singleShot(100, loop.quit)
            loop.exec()
            return


def test_gui_workflow(app, tmp_path, portrait_like):
    from PIL import Image

    from picme.gui.app import MainWindow

    paths = []
    for name in ("a.jpg", "b.png"):
        p = tmp_path / name
        Image.fromarray(portrait_like[..., ::-1]).save(p)
        paths.append(p)

    win = MainWindow(analyzer=FakeAnalyzer())
    win.add_paths(paths)
    _wait(win)
    assert win.current is win.entries[0]
    assert win.person_combo.count() == 3  # 全員 + 2 人
    assert win.view.after is not None

    # 共通設定の変更
    win.sliders[("skin", "smooth")][0].setValue(66)
    assert win.global_settings.skin.smooth == 66

    # 個別設定: 共通設定には影響しない
    win.chk_own.setChecked(True)
    win.sliders[("face", "slim")][0].setValue(33)
    assert win.current.own.face.slim == 33 and win.global_settings.face.slim != 33
    assert "★" in win.list.item(0).text()

    # 人物ごとの設定
    win.person_combo.setCurrentIndex(2)
    win.sliders[("face", "eye_enlarge")][0].setValue(50)
    assert len(win.current.overrides) == 1 and win.current.overrides[0].x > 0.5
    assert win.person_clear.isEnabled()

    # 修正ブラシ
    win._add_spot(0.5, 0.5, 0.01)
    assert win.effective(win.current).heal_spots == [[0.5, 0.5, 0.01]]
    _wait(win)

    # 元に戻す: スポット → 人物設定 → 個別設定 の順に戻る
    win._flush_history()
    win.undo()
    assert win.current.spots == []
    win.undo()
    assert win.current.overrides == []
    win.redo()
    assert len(win.current.overrides) == 1

    # 別の写真は共通設定
    win.list.setCurrentRow(1)
    _wait(win)
    assert win.target_settings() is win.global_settings
    assert win.sliders[("skin", "smooth")][0].value() == 66
    win.close()
