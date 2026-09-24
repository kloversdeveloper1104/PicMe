"""配布用アプリ (Windows: PicMe.exe フォルダ / macOS: PicMe.app / Linux) をビルドする。

    pip install -e ".[all]" pyinstaller
    python packaging/build.py

成果物は dist/PicMe (macOS は dist/PicMe.app) 。AI モデルも同梱される。
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    sys.path.insert(0, str(ROOT))
    from picme.models import MODEL_URLS, ensure_model

    models = ROOT / "models"
    models.mkdir(exist_ok=True)
    for name in MODEL_URLS:
        path = ensure_model(name)
        if path is None:
            print(f"モデル {name} を取得できませんでした", file=sys.stderr)
            return 1
        if path.parent != models:
            shutil.copy2(path, models / name)
    cmd = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", str(ROOT / "packaging" / "picme.spec")]
    print(" ".join(cmd))
    return subprocess.call(cmd, cwd=ROOT)


if __name__ == "__main__":
    sys.exit(main())
