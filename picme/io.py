"""画像の読み書き。EXIF の回転を反映し、書き出し時は EXIF / ICC プロファイルを引き継ぐ。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

RASTER_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}
HEIF_EXTS = {".heic", ".heif", ".hif"}
RAW_EXTS = {
    ".cr2", ".cr3", ".crw", ".nef", ".nrw", ".arw", ".srf", ".sr2", ".raf", ".orf", ".rw2",
    ".pef", ".dng", ".3fr", ".iiq", ".erf", ".mef", ".mos", ".x3f", ".srw", ".kdc",
}


def _heif_available() -> bool:
    try:
        from pillow_heif import register_heif_opener
    except ImportError:
        return False
    register_heif_opener()
    return True


def _raw_available() -> bool:
    try:
        import rawpy  # noqa: F401
    except ImportError:
        return False
    return True


# オプションのライブラリが入っている場合だけ HEIC / RAW を扱う
SUPPORTED_EXTS = RASTER_EXTS | (HEIF_EXTS if _heif_available() else set()) | (RAW_EXTS if _raw_available() else set())
# 書き出し時に元の拡張子を使えない形式 (JPEG で書き出す)
READ_ONLY_EXTS = HEIF_EXTS | RAW_EXTS


@dataclass
class ImageMeta:
    exif: bytes | None = None
    icc_profile: bytes | None = None


def is_image(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_EXTS


def _load_raw(path: Path) -> tuple[np.ndarray, ImageMeta]:
    import rawpy

    with rawpy.imread(str(path)) as raw:
        rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=False, output_bps=8, user_flip=None)
    meta = ImageMeta()
    try:  # 撮影情報は埋め込みプレビューの EXIF から取れることが多い
        with Image.open(path) as im:
            exif = im.getexif()
            if exif:
                exif[0x0112] = 1
                meta.exif = exif.tobytes()
    except Exception:
        pass
    return np.ascontiguousarray(rgb[..., ::-1]), meta


def load_image(path: str | Path) -> tuple[np.ndarray, ImageMeta]:
    """BGR uint8 配列とメタ情報を返す。RAW は現像 (カメラの WB) してから返す。"""
    path = Path(path)
    if path.suffix.lower() in RAW_EXTS:
        return _load_raw(path)
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
