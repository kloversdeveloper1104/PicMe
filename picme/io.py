"""画像の読み書き。EXIF の回転を反映し、書き出し時は EXIF / ICC プロファイルを引き継ぐ。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}


@dataclass
class ImageMeta:
    exif: bytes | None = None
    icc_profile: bytes | None = None


def is_image(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_EXTS


def load_image(path: str | Path) -> tuple[np.ndarray, ImageMeta]:
    """BGR uint8 配列とメタ情報を返す。"""
    with Image.open(path) as im:
        exif = im.getexif()
        im = ImageOps.exif_transpose(im)
        meta = ImageMeta(icc_profile=im.info.get("icc_profile"))
        if exif:
            exif[0x0112] = 1  # 回転は画素に反映済み
            meta.exif = exif.tobytes()
        rgb = np.asarray(im.convert("RGB"))
    return np.ascontiguousarray(rgb[..., ::-1]), meta


def save_image(path: str | Path, bgr: np.ndarray, meta: ImageMeta | None = None, quality: int = 95) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.fromarray(np.ascontiguousarray(bgr[..., ::-1]))
    kwargs: dict = {}
    ext = path.suffix.lower()
    if meta:
        if meta.icc_profile:
            kwargs["icc_profile"] = meta.icc_profile
        if meta.exif and ext in {".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".png"}:
            kwargs["exif"] = meta.exif
    if ext in {".jpg", ".jpeg"}:
        kwargs.update(quality=quality, subsampling=0, optimize=True)
    elif ext == ".webp":
        kwargs.update(quality=quality)
    im.save(path, **kwargs)
