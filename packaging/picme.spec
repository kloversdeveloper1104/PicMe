# PyInstaller 設定。packaging/build.py から使う。
# -*- mode: python ; coding: utf-8 -*-
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all

ROOT = Path(SPECPATH).parent
datas, binaries, hiddenimports = [], [], []
for pkg in ("mediapipe",):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h
for opt in ("rawpy", "pillow_heif"):
    try:
        __import__(opt)
    except ImportError:
        continue
    d, b, h = collect_all(opt)
    datas += d
    binaries += b
    hiddenimports += h

# AI モデルを同梱して、初回起動時のダウンロードを不要にする
models = ROOT / "models"
datas += [(str(p), "models") for p in models.glob("*") if p.suffix in {".task", ".tflite"}]

a = Analysis(
    [str(ROOT / "packaging" / "launcher.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports + ["picme.gui.app"],
    excludes=["tkinter", "PySide6.QtWebEngineCore", "PySide6.Qt3DCore"],
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="PicMe",
    console=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="PicMe")
if sys.platform == "darwin":
    app = BUNDLE(coll, name="PicMe.app", bundle_identifier="app.picme.retouch")
