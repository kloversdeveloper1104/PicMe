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
    wrinkles: float = 0.0  # 0..100 しわ・ほうれい線の軽減


@dataclass
class FaceSettings:
    slim: float = 0.0  # 0..100 輪郭(小顔)
    eye_enlarge: float = 0.0  # 0..100 デカ目
    eye_brighten: float = 0.0  # 0..100 目の明るさ・クリア感
    dark_circles: float = 0.0  # 0..100 クマ除去
    teeth_whiten: float = 0.0  # 0..100 歯のホワイトニング
    lip_color: float = 0.0  # 0..100 唇の血色
    hair_shine: float = 0.0  # 0..100 髪のツヤ


@dataclass
class BackgroundSettings:
    blur: float = 0.0  # 0..100 背景ぼかし
    replace_color: str = ""  # 背景を塗りつぶす色 (#RRGGBB)。空なら置き換えない
    replace_image: str = ""  # 背景に使う画像のパス。replace_color より優先


@dataclass
class FaceOverride:
    """特定の人物 (顔) だけに適用する設定。顔は画像内の位置 (正規化座標) で識別する。"""

    x: float  # 顔の中心 (0..1, 画像幅に対する割合)
    y: float
    face: FaceSettings = field(default_factory=FaceSettings)
    skin: float = 100.0  # この人の美肌の強さ (0..100 %)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FaceOverride":
        return cls(x=float(d["x"]), y=float(d["y"]), face=_load_section(FaceSettings(), d.get("face") or {}), skin=float(d.get("skin", 100.0)))


@dataclass
class RetouchSettings:
    color: ColorSettings = field(default_factory=ColorSettings)
    skin: SkinSettings = field(default_factory=SkinSettings)
    face: FaceSettings = field(default_factory=FaceSettings)
    background: BackgroundSettings = field(default_factory=BackgroundSettings)
    # ---- 画像ごとのデータ (プリセットとしては保存しない) ----
    face_overrides: list[FaceOverride] = field(default_factory=list)
    # 修正ブラシで指定した箇所 [x, y, r] (x, y は 0..1、r は画像の長辺に対する割合)
    heal_spots: list[list[float]] = field(default_factory=list)

    SECTIONS = ("color", "skin", "face", "background")

    # ---- シリアライズ -------------------------------------------------
    def to_dict(self, include_image_data: bool = True) -> dict[str, Any]:
        d = {name: asdict(getattr(self, name)) for name in self.SECTIONS}
        if include_image_data:
            d["face_overrides"] = [o.to_dict() for o in self.face_overrides]
            d["heal_spots"] = [list(map(float, s)) for s in self.heal_spots]
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RetouchSettings":
        """未知のキーは無視し、欠けているキーはデフォルト値で補う。"""
        settings = cls()
        for name in cls.SECTIONS:
            _load_section(getattr(settings, name), data.get(name) or {})
        settings.face_overrides = [FaceOverride.from_dict(o) for o in data.get("face_overrides") or []]
        settings.heal_spots = [[float(v) for v in s[:3]] for s in data.get("heal_spots") or [] if len(s) >= 3]
        return settings

    def preset_only(self) -> "RetouchSettings":
        """画像固有のデータ (人物ごとの設定・修正箇所) を除いたコピー。"""
        return RetouchSettings.from_dict(self.to_dict(include_image_data=False))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(include_image_data=False), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "RetouchSettings":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def copy(self) -> "RetouchSettings":
        return RetouchSettings.from_dict(self.to_dict())


def _load_section(section, values: dict[str, Any]):
    for f in fields(section):
        if f.name not in values:
            continue
        default, value = getattr(section, f.name), values[f.name]
        if isinstance(default, bool):
            value = bool(value)
        elif isinstance(default, str):
            value = "" if value is None else str(value)
        else:
            value = float(value)
        setattr(section, f.name, value)
    return section


def _preset(**groups: dict[str, Any]) -> RetouchSettings:
    return RetouchSettings.from_dict(groups)


# 組み込みプリセット (名前 -> 設定)。GUI のプルダウンや CLI の --preset で使う。
PRESETS: dict[str, RetouchSettings] = {
    "none": _preset(color={"auto_white_balance": False, "auto_tone": False}),
    "natural": _preset(
        color={"vibrance": 10, "sharpen": 15},
        skin={"smooth": 35, "texture": 60, "blemish": 60, "even_tone": 30, "brighten": 10, "wrinkles": 15},
        face={"eye_brighten": 20, "dark_circles": 35, "teeth_whiten": 30, "hair_shine": 10},
    ),
    "portrait": _preset(
        color={"contrast": 10, "shadows": 15, "highlights": -15, "vibrance": 15, "sharpen": 20},
        skin={"smooth": 55, "texture": 45, "blemish": 80, "even_tone": 45, "brighten": 15, "wrinkles": 35},
        face={"slim": 20, "eye_enlarge": 10, "eye_brighten": 35, "dark_circles": 55, "teeth_whiten": 45, "lip_color": 15, "hair_shine": 20},
    ),
    "wedding": _preset(
        color={"exposure": 8, "contrast": -5, "highlights": -25, "shadows": 25, "temperature": 8, "vibrance": 10, "sharpen": 10},
        skin={"smooth": 65, "texture": 40, "blemish": 90, "even_tone": 55, "brighten": 25, "wrinkles": 45},
        face={"slim": 25, "eye_enlarge": 12, "eye_brighten": 40, "dark_circles": 65, "teeth_whiten": 55, "lip_color": 20, "hair_shine": 30},
        background={"blur": 25},
    ),
    "studio": _preset(
        color={"contrast": 15, "highlights": -10, "shadows": 5, "vibrance": 5, "sharpen": 30},
        skin={"smooth": 45, "texture": 55, "blemish": 85, "even_tone": 50, "brighten": 5, "wrinkles": 30},
        face={"slim": 10, "eye_brighten": 30, "dark_circles": 50, "teeth_whiten": 40},
    ),
    "men": _preset(
        color={"contrast": 15, "shadows": 10, "vibrance": 5, "saturation": -5, "sharpen": 35},
        skin={"smooth": 20, "texture": 80, "blemish": 70, "even_tone": 25, "wrinkles": 20},
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
