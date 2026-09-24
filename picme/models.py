"""MediaPipe モデルファイルの取得・キャッシュ。"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path

log = logging.getLogger(__name__)

MODEL_URLS = {
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/1/face_landmarker.task"
    ),
    "selfie_multiclass_256x256.tflite": (
        "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
        "selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"
    ),
}


def model_dirs() -> list[Path]:
    """モデルを探すディレクトリ (優先順)。"""
    dirs = []
    if env := os.environ.get("PICME_MODEL_DIR"):
        dirs.append(Path(env))
    dirs.append(Path(__file__).resolve().parent.parent / "models")
    dirs.append(Path.home() / ".cache" / "picme" / "models")
    return dirs


def find_model(name: str) -> Path | None:
    for d in model_dirs():
        p = d / name
        if p.is_file() and p.stat().st_size > 0:
            return p
    return None


def ensure_model(name: str, download: bool = True) -> Path | None:
    """モデルのパスを返す。無ければダウンロードを試み、失敗したら None。"""
    if path := find_model(name):
        return path
    if not download or os.environ.get("PICME_OFFLINE"):
        return None
    target_dir = model_dirs()[-1]
    target = target_dir / name
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        log.info("モデルをダウンロード中: %s", name)
        with urllib.request.urlopen(MODEL_URLS[name], timeout=60) as resp, tempfile.NamedTemporaryFile(
            dir=target_dir, delete=False
        ) as tmp:
            shutil.copyfileobj(resp, tmp)
        Path(tmp.name).replace(target)
        return target
    except Exception as exc:  # ネットワーク不可など。AI 機能なしで動作を続ける
        log.warning("モデル %s を取得できませんでした: %s", name, exc)
        return None


def download_all() -> dict[str, Path | None]:
    return {name: ensure_model(name) for name in MODEL_URLS}
