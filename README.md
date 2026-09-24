# PicMe — AI ポートレート自動レタッチ

EVOTO AI のように、人物写真を **AI で解析して自動で美肌・顔補正・色補正** し、
同じ設定で **大量の写真を一括処理** できるソフトです。

- 顔ランドマーク (478 点) と肌/人物セグメンテーションは Google MediaPipe を使用
- すべてローカルで処理します (写真を外部に送信しません)
- GUI (デスクトップアプリ) と CLI (コマンドライン一括処理) の両方に対応

## 主な機能

| 分類 | 機能 |
| --- | --- |
| 基本補正 | 自動ホワイトバランス / 自動トーン / 露出 / コントラスト / ハイライト / シャドウ / 色温度 / 色かぶり / 自然な彩度 / 彩度 / シャープ |
| 美肌 | なめらかさ (周波数分離で**質感を残す**) / シミ・ニキビ除去 / 肌色ムラ補正 / 肌の明るさ |
| 顔・パーツ | 小顔 (輪郭) / デカ目 / 目の明るさ・白目の充血除去 / クマ除去 / 歯のホワイトニング / 唇の血色 |
| 背景 | 背景ぼかし (人物を自動で切り抜き) |
| ワークフロー | プリセット (ナチュラル / ポートレート / ウェディング / スタジオ / メンズ)・自作プリセットの保存と読込 (JSON)・フォルダ一括書き出し (マルチプロセス)・EXIF / ICC プロファイルの引き継ぎ |

複数人が写った写真では、検出したすべての顔に補正がかかります。
処理の強さは顔の大きさに合わせて自動調整されるため、解像度に関係なく同じ仕上がりになります。

## インストール

Python 3.10 以上が必要です。

```bash
pip install -e ".[gui]"
# 初回起動時に AI モデル (約 20MB) を自動ダウンロードします。事前に取得する場合:
picme --download-models
```

Linux で `libEGL.so.1` が無いというエラーが出る場合は `sudo apt install libegl1 libgles2` を実行してください。
AI モデルが使えない環境でも、色補正と色ベースの簡易美肌で動作します (`--no-ai`)。

## 使い方

### GUI

```bash
picme-gui            # または python -m picme.gui.app
picme-gui 写真フォルダ/
```

1. 「画像を開く」「フォルダを開く」またはドラッグ&ドロップで写真を読み込む
2. 右側でプリセットを選び、スライダーで好みに調整 (すべての写真に共通で適用)
3. 中央のビューで白い境界線をドラッグして補正前 / 補正後を比較
4. 「この画像を書き出し」または「一括書き出し」で保存
5. 調整した設定は「プリセット保存」で JSON に保存でき、次回や CLI でも使えます

### CLI (一括処理)

```bash
# フォルダ内の写真をポートレート設定で一括補正
picme 撮影データ/ -o 納品/ --preset portrait

# 自作プリセット + 個別の上書き、サブフォルダも処理、PNG で出力
picme 撮影データ/ -r -o 納品/ --preset my_preset.json --set skin.smooth=70 --set face.slim=30 --format png

# プリセット一覧
picme --list-presets
```

主なオプション: `-j` 並列数 / `--suffix _retouched` 出力名の接尾辞 / `-q` JPEG 品質 / `--skip-existing` 出力済みをスキップ /
`--save-preset FILE` 現在の設定を保存。

### Python から使う

```python
from picme import Processor, get_preset
from picme.io import load_image, save_image

img, meta = load_image("photo.jpg")
settings = get_preset("wedding")
settings.face.slim = 30
out, analysis = Processor().process(img, settings)
save_image("photo_retouched.jpg", out, meta)
print(len(analysis.faces), "人の顔を補正しました")
```

## 設定項目

プリセット JSON は次の形式です (値は 0〜100、基本補正の多くは -100〜100)。

```json
{
  "color": {"auto_white_balance": true, "auto_tone": true, "exposure": 0, "contrast": 10,
            "highlights": -15, "shadows": 15, "temperature": 0, "tint": 0,
            "vibrance": 15, "saturation": 0, "sharpen": 20},
  "skin": {"smooth": 55, "texture": 45, "blemish": 80, "even_tone": 45, "brighten": 15},
  "face": {"slim": 20, "eye_enlarge": 10, "eye_brighten": 35, "dark_circles": 55,
           "teeth_whiten": 45, "lip_color": 15},
  "background": {"blur": 0}
}
```

## 処理の仕組み

1. **AI 解析** (`picme/analysis.py`): MediaPipe Face Landmarker で顔の 478 点を検出し、
   Selfie Multiclass セグメンテーションで肌・人物の領域を推定。目・眉・唇は肌マスクから除外
2. **基本補正** (`retouch/color.py`): 肌以外の領域から色かぶりを推定する Shades-of-Gray 法のホワイトバランス、黒点/白点とガンマの自動調整など
3. **美肌** (`retouch/skin.py`): 周囲より暗い/赤い小さな斑点を検出して周辺の肌で埋め、
   Guided Filter で凹凸をならした上に元の高周波 (肌理) を戻す周波数分離
4. **顔パーツ** (`retouch/face.py`): ランドマークから目・目の下・口のマスクを作り局所補正
5. **背景** (`retouch/background.py`): 人物マスクで前景を保護しながら背景だけをぼかす
6. **変形**: 小顔・デカ目は変位場を合成して最後に 1 回だけ再サンプリング (歪み・二重変形を防止)

GUI のプレビューは縮小画像で解析結果をキャッシュして高速に表示し、書き出し時はフル解像度で再解析します。

## 開発

```bash
pip install -e ".[gui,dev]"
pytest
```

## ライセンスについて

AI モデルは Google MediaPipe のモデル (Apache License 2.0) を初回実行時にダウンロードして使用します。
