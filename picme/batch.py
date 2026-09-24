"""フォルダ単位の一括自動補正 (マルチプロセス対応)。"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from .io import READ_ONLY_EXTS, is_image, load_image, save_image
from .settings import RetouchSettings

log = logging.getLogger(__name__)


@dataclass
class BatchResult:
    source: Path
    output: Path | None
    faces: int = 0
    error: str | None = None


def collect_images(inputs: Iterable[str | Path], recursive: bool = False) -> list[Path]:
    files: list[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            it = p.rglob("*") if recursive else p.iterdir()
            files.extend(sorted(f for f in it if f.is_file() and is_image(f)))
        elif p.is_file() and is_image(p):
            files.append(p)
    return files


def output_path(src: Path, out_dir: Path, suffix: str = "", fmt: str | None = None) -> Path:
    if fmt:
        ext = f".{fmt.lstrip('.')}"
    elif src.suffix.lower() in READ_ONLY_EXTS:
        ext = ".jpg"  # RAW / HEIC は JPEG で書き出す
    else:
        ext = src.suffix
    return out_dir / f"{src.stem}{suffix}{ext}"


# ---- ワーカープロセス側 ---------------------------------------------------
_processor = None


def _init_worker(use_ai: bool) -> None:
    global _processor
    import cv2

    from .pipeline import Processor

    cv2.setNumThreads(1)  # プロセス並列と OpenCV のスレッドが競合しないように
    _processor = Processor(use_ai=use_ai)


def _process_one(src: Path, dst: Path, settings_dict: dict, quality: int) -> BatchResult:
    try:
        img, meta = load_image(src)
        out, analysis = _processor.process(img, RetouchSettings.from_dict(settings_dict))
        save_image(dst, out, meta, quality)
        return BatchResult(src, dst, len(analysis.faces))
    except Exception as exc:  # 1 枚の失敗で全体を止めない
        return BatchResult(src, None, error=f"{type(exc).__name__}: {exc}")


def run_batch(
    files: list[Path],
    out_dir: Path,
    settings: RetouchSettings,
    *,
    workers: int | None = None,
    suffix: str = "",
    fmt: str | None = None,
    quality: int = 95,
    overwrite: bool = True,
    use_ai: bool = True,
    progress: Callable[[int, int, BatchResult], None] | None = None,
    mp_context=None,
    per_file: dict[Path, RetouchSettings] | None = None,
) -> list[BatchResult]:
    """files を settings で一括補正する。per_file に指定した画像はその設定を優先する。"""
    out_dir = Path(out_dir)
    jobs = []
    for src in files:
        dst = output_path(src, out_dir, suffix, fmt)
        if dst.resolve() == src.resolve():
            raise ValueError(f"出力先が元画像と同じです: {src}")
        if not overwrite and dst.exists():
            continue
        jobs.append((src, dst))

    workers = workers or max(1, min(len(jobs), (os.cpu_count() or 2) // 2))
    per_file = per_file or {}
    base_dict = settings.to_dict()
    job_settings = {src: (per_file[src].to_dict() if src in per_file else base_dict) for src, _ in jobs}
    results: list[BatchResult] = []
    total = len(jobs)

    if workers <= 1:
        _init_worker(use_ai)
        for src, dst in jobs:
            r = _process_one(src, dst, job_settings[src], quality)
            results.append(r)
            if progress:
                progress(len(results), total, r)
        return results

    with ProcessPoolExecutor(
        max_workers=workers, mp_context=mp_context, initializer=_init_worker, initargs=(use_ai,)
    ) as ex:
        futures = [ex.submit(_process_one, s, d, job_settings[s], quality) for s, d in jobs]
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            if progress:
                progress(len(results), total, r)
    return results
