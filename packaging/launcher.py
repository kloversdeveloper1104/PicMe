"""配布版 (PyInstaller) の起動スクリプト。

引数に画像やフォルダが渡された場合は GUI で開く。
`PicMe --cli ...` とするとコマンドライン版として動く。
"""

import multiprocessing
import os
import sys


def _ensure_streams() -> None:
    """ウィンドウアプリとしてビルドすると stdout/stderr が無いので用意する。"""
    if sys.stdout is not None:
        return
    stream = None
    if sys.platform == "win32":
        import ctypes

        if ctypes.windll.kernel32.AttachConsole(-1):  # 起動元のコマンドプロンプトに出力する
            stream = open("CONOUT$", "w", encoding="utf-8")
    sys.stdout = sys.stderr = stream or open(os.devnull, "w", encoding="utf-8")


if __name__ == "__main__":
    multiprocessing.freeze_support()  # 一括書き出しのワーカープロセス用 (Windows / macOS で必須)
    _ensure_streams()
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        from picme.cli import main as cli_main

        sys.exit(cli_main(sys.argv[2:]))
    from picme.gui.app import main

    sys.exit(main())
