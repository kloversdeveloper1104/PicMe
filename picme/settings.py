"""レタッチ設定とプリセット。

すべてのパラメータは 0〜100 (または -100〜100) の「スライダー値」で保持し、
実際の処理強度への変換は各処理モジュール側で行う。これにより GUI・CLI・
JSON プリセットで同じ値をそのまま扱える。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any


@dataclass
class ColorSettings:
    auto_white_balance: bool = True
    auto_tone: bool = True
    exposure: float = 0.0  # -100..100 (±2EV)
    contrast: float = 0.0  # -100..100
    highlights: float = 0.0  # -100..100 (マイナスで白飛び抑制)
    shadows: float = 0.0  # -100..100 (プラスで暗部持ち上げ)
    temperature: float = 0.0  # -100..100 (プラスで暖色)
    tint: float = 0.0  # -100..100 (プラスでマゼンタ)
    vibrance: float = 0.0  # -100..100
    saturation: float = 0.0  # -100..100
    sharpen: float = 0.0  # 0..100


@dataclass
class SkinSettings:
    smooth: float = 0.0  # 0..100 肌のなめらかさ
    texture: float = 50.0  # 0..100 残す毛穴・質感の量
    blemish: float = 0.0  # 0..100 シミ・ニキビ除去
    even_tone: float = 0.0  # 0..100 肌色ムラ補正
    brighten: float = 0.0  # 0..100 肌の明るさ


@dataclass
class FaceSettings:
    slim: float = 0.0  # 0..100 輪郭(小顔)
    eye_enlarge: float = 0.0  # 0..100 デカ目
    eye_brighten: float = 0.0  # 0..100 目の明るさ・クリア感
    dark_circles: float = 0.0  # 0..100 クマ除去
    teeth_whiten: float = 0.0  # 0..100 歯のホワイトニング
    lip_color: float = 0.0  # 0..100 唇の血色


@dataclass
class BackgroundSettings:
    blur: float = 0.0  # 0..100 背景ぼかし


@dataclass
class RetouchSettings:
    color: ColorSettings = field(default_factory=ColorSettings)
    skin: SkinSettings = field(default_factory=SkinSettings)
    face: FaceSettings = field(default_factory=FaceSettings)
    background: BackgroundSettings = field(default_factory=BackgroundSettings)

    # ---- シリアライズ -------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RetouchSettings":
        """未知のキーは無視し、欠けているキーはデフォルト値で補う。"""
        settings = cls()
        for group in fields(cls):
            section = getattr(settings, group.name)
            values = data.get(group.name) or {}
            for f in fields(section):
                if f.name in values:
                    default = getattr(section, f.name)
                    value = values[f.name]
                    setattr(section, f.name, bool(value) if isinstance(default, bool) else float(value))
        return settings

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "RetouchSettings":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def copy(self) -> "RetouchSettings":
        return RetouchSettings.from_dict(self.to_dict())


def _preset(**groups: dict[str, Any]) -> RetouchSettings:
    return RetouchSettings.from_dict(groups)


# 組み込みプリセット (名前 -> 設定)。GUI のプルダウンや CLI の --preset で使う。
PRESETS: dict[str, RetouchSettings] = {
    "none": _preset(color={"auto_white_balance": False, "auto_tone": False}),
    "natural": _preset(
        color={"vibrance": 10, "sharpen": 15},
        skin={"smooth": 35, "texture": 60, "blemish": 60, "even_tone": 30, "brighten": 10},
        face={"eye_brighten": 20, "dark_circles": 35, "teeth_whiten": 30},
    ),
    "portrait": _preset(
        color={"contrast": 10, "shadows": 15, "highlights": -15, "vibrance": 15, "sharpen": 20},
        skin={"smooth": 55, "texture": 45, "blemish": 80, "even_tone": 45, "brighten": 15},
        face={"slim": 20, "eye_enlarge": 10, "eye_brighten": 35, "dark_circles": 55, "teeth_whiten": 45, "lip_color": 15},
    ),
    "wedding": _preset(
        color={"exposure": 8, "contrast": -5, "highlights": -25, "shadows": 25, "temperature": 8, "vibrance": 10, "sharpen": 10},
        skin={"smooth": 65, "texture": 40, "blemish": 90, "even_tone": 55, "brighten": 25},
        face={"slim": 25, "eye_enlarge": 12, "eye_brighten": 40, "dark_circles": 65, "teeth_whiten": 55, "lip_color": 20},
        background={"blur": 25},
    ),
    "studio": _preset(
        color={"contrast": 15, "highlights": -10, "shadows": 5, "vibrance": 5, "sharpen": 30},
        skin={"smooth": 45, "texture": 55, "blemish": 85, "even_tone": 50, "brighten": 5},
        face={"slim": 10, "eye_brighten": 30, "dark_circles": 50, "teeth_whiten": 40},
    ),
    "men": _preset(
        color={"contrast": 15, "shadows": 10, "vibrance": 5, "saturation": -5, "sharpen": 35},
        skin={"smooth": 20, "texture": 80, "blemish": 70, "even_tone": 25},
        face={"eye_brighten": 15, "dark_circles": 40, "teeth_whiten": 30},
    ),
}

PRESET_LABELS: dict[str, str] = {
    "none": "補正なし",
    "natural": "ナチュラル",
    "portrait": "ポートレート",
    "wedding": "ウェディング / ブライダル",
    "studio": "スタジオ・証明写真",
    "men": "メンズ",
}


def get_preset(name_or_path: str) -> RetouchSettings:
    """組み込みプリセット名、または JSON ファイルのパスから設定を得る。"""
    if name_or_path in PRESETS:
        return PRESETS[name_or_path].copy()
    path = Path(name_or_path)
    if path.is_file():
        return RetouchSettings.load(path)
    raise KeyError(f"プリセットが見つかりません: {name_or_path} (利用可能: {', '.join(PRESETS)})")
