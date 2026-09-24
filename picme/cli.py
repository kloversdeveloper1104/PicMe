"""コマンドライン: 画像・フォルダを一括で自動補正する。

例:
    picme 写真フォルダ/ -o 出力/ --preset portrait
    picme a.jpg b.jpg -o out/ --preset my_preset.json --set skin.smooth=70 --set face.slim=30
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from .batch import collect_images, run_batch
from .settings import PRESET_LABELS, PRESETS, RetouchSettings, get_preset


def _apply_overrides(settings: RetouchSettings, overrides: list[str]) -> RetouchSettings:
    data = settings.to_dict()
    for item in overrides:
        try:
            key, value = item.split("=", 1)
            group, name = key.strip().split(".", 1)
        except ValueError:
            raise SystemExit(f"--set の形式が不正です (例: skin.smooth=60): {item}")
        if group not in data or name not in data[group]:
            raise SystemExit(f"不明な設定項目です: {key}")
        v = value.strip().lower()
        data[group][name] = v in {"1", "true", "on", "yes"} if isinstance(data[group][name], bool) else float(v)
    return RetouchSettings.from_dict(data)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="picme", description="AI ポートレート自動レタッチ (一括処理)")
    p.add_argument("inputs", nargs="*", help="画像ファイルまたはフォルダ")
    p.add_argument("-o", "--output", default="output", help="出力フォルダ (既定: ./output)")
    p.add_argument("-p", "--preset", default="natural", help="プリセット名または JSON ファイル")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="GROUP.KEY=VALUE",
                   help="設定を個別に上書き (複数指定可)。例: --set skin.smooth=60")
    p.add_argument("-r", "--recursive", action="store_true", help="サブフォルダも処理する")
    p.add_argument("-j", "--workers", type=int, default=None, help="並列プロセス数")
    p.add_argument("--suffix", default="", help="出力ファイル名の接尾辞 (例: _retouched)")
    p.add_argument("--format", choices=["jpg", "png", "tif", "webp"], help="出力形式 (既定: 入力と同じ)")
    p.add_argument("-q", "--quality", type=int, default=95, help="JPEG/WebP 品質 (既定 95)")
    p.add_argument("--skip-existing", action="store_true", help="出力済みの画像はスキップ")
    p.add_argument("--no-ai", action="store_true", help="AI モデルを使わない (色補正と簡易美肌のみ)")
    p.add_argument("--save-preset", metavar="FILE", help="最終的な設定を JSON に保存")
    p.add_argument("--list-presets", action="store_true", help="組み込みプリセットを一覧表示")
    p.add_argument("--download-models", action="store_true", help="AI モデルを事前にダウンロード")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s: %(message)s")

    if args.list_presets:
        for name in PRESETS:
            print(f"{name:10s} {PRESET_LABELS.get(name, '')}")
        return 0
    if args.download_models:
        from .models import download_all

        for name, path in download_all().items():
            print(f"{name}: {path or '取得失敗'}")
        return 0

    try:
        settings = _apply_overrides(get_preset(args.preset), args.overrides)
    except KeyError as exc:
        print(exc.args[0], file=sys.stderr)
        return 2
    if args.save_preset:
        settings.save(args.save_preset)
        print(f"設定を保存しました: {args.save_preset}")

    if not args.inputs:
        if not args.save_preset:
            build_parser().print_help()
        return 0
    files = collect_images(args.inputs, args.recursive)
    if not files:
        print("処理できる画像が見つかりません", file=sys.stderr)
        return 1

    out_dir = Path(args.output)
    print(f"{len(files)} 枚を処理します → {out_dir}/  (プリセット: {args.preset})")
    start = time.time()

    def progress(done: int, total: int, r) -> None:
        status = f"顔 {r.faces}" if r.error is None else f"エラー: {r.error}"
        print(f"[{done}/{total}] {r.source.name}  {status}", flush=True)

    results = run_batch(
        files, out_dir, settings,
        workers=args.workers, suffix=args.suffix, fmt=args.format, quality=args.quality,
        overwrite=not args.skip_existing, use_ai=not args.no_ai, progress=progress,
    )
    failed = [r for r in results if r.error]
    print(f"完了: {len(results) - len(failed)} 枚成功 / {len(failed)} 枚失敗  ({time.time() - start:.1f} 秒)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
