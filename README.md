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
| 美肌 | なめらかさ (周波数分離で**質感を残す**) / シミ・ニキビ除去 / しわ・ほうれい線の軽減 / 肌色ムラ補正 / 肌の明るさ |
| 顔・パーツ | 小顔 (輪郭) / デカ目 / 目の明るさ・白目の充血除去 / クマ除去 / 歯のホワイトニング / 唇の血色 / 髪のツヤ |
| 背景 | 背景ぼかし / 背景の置き換え (単色・画像) — 人物は自動で切り抜き |
| 手動修正 | 修正ブラシ (なぞった箇所を周囲の肌の質感で修復) |
| 人物ごとの補正 | 集合写真で「この人だけ小顔を弱める」「この人は美肌 50%」など |
| ワークフロー | 全写真共通の設定 + 写真ごとの個別設定・元に戻す/やり直し・プリセット (ナチュラル / ポートレート / ウェディング / スタジオ / メンズ) と自作プリセット (JSON)・フォルダ一括書き出し (マルチプロセス)・EXIF / ICC プロファイルの引き継ぎ |
| 対応形式 | JPEG / PNG / TIFF / WebP / BMP、HEIC (iPhone)、カメラの RAW (CR2/CR3/NEF/ARW/RAF/ORF/RW2/DNG など) |

複数人が写った写真では、検出したすべての顔に補正がかかります (集合写真の小さな顔も検出します)。
処理の強さは顔の大きさに合わせて自動調整されるため、解像度に関係なく同じ仕上がりになります。

## インストール

### 配布版アプリ (Python 不要)

GitHub の Releases (または Actions の「Build apps」の成果物) から、お使いの OS 用の zip をダウンロードして展開し、
`PicMe` (Windows は `PicMe.exe`、macOS は `PicMe.app`) を起動してください。AI モデルも同梱されています。
コマンドライン版は `PicMe --cli 写真フォルダ/ -o 出力/` で使えます。

### Python から

Python 3.10 以上が必要です。

```bash
pip install -e ".[all]"     # GUI・RAW・HEIC 対応をまとめて入れる (最小構成は pip install -e ".[gui]")
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
3. 特定の写真だけ変えたいときは「この写真だけ個別に調整」をオン (一覧に ★ が付きます)
4. 集合写真では「対象」で人物を選ぶか、写真上の番号をクリックすると、その人だけの設定にできます
5. 取り切れないシミやホクロは「修正ブラシ」でなぞって消せます (一覧に ✎ が付きます)
6. 中央のビューで白い境界線をドラッグして補正前 / 補正後を比較。ホイールで拡大、右ドラッグで移動
7. 「この画像を書き出し」または「一括書き出し」で保存
8. 調整した設定は「プリセット保存」で JSON に保存でき、次回や CLI でも使えます

| 操作 | ショートカット |
| --- | --- |
| 元に戻す / やり直し | Ctrl+Z / Ctrl+Shift+Z (Ctrl+Y) |
| 全体表示 / 等倍 (100%) | Ctrl+0 / Ctrl+1 |
| 表示ツール / 修正ブラシ | V / B |
| 表示位置の移動 | 右ドラッグ、中ドラッグ、Space+ドラッグ |
| この画像を書き出し / 一括書き出し | Ctrl+E / Ctrl+Shift+E |

「高画質プレビュー」をオンにすると元の解像度でプレビューします (肌の質感を等倍で確認するとき用)。

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
  "skin": {"smooth": 55, "texture": 45, "blemish": 80, "wrinkles": 35, "even_tone": 45, "brighten": 15},
  "face": {"slim": 20, "eye_enlarge": 10, "eye_brighten": 35, "dark_circles": 55,
           "teeth_whiten": 45, "lip_color": 15, "hair_shine": 20},
  "background": {"blur": 0, "replace_color": "", "replace_image": ""}
}
```

`replace_color` に `"#ffffff"` のような色、または `replace_image` に画像のパスを入れると背景を置き換えます
(CLI では `picme 証明写真/ --preset studio --set "background.replace_color=#e8eef5"` のように指定できます)。

## 処理の仕組み

1. **AI 解析** (`picme/analysis.py`): MediaPipe Face Landmarker で顔の 478 点を検出し
   (画像全体 + 重なりのあるタイルで検出して集合写真の小さな顔も拾う)、
   Selfie Multiclass セグメンテーションで肌・髪・人物の領域を推定。目・眉・唇は肌マスクから除外
2. **基本補正** (`retouch/color.py`): 肌以外の領域から色かぶりを推定する Shades-of-Gray 法のホワイトバランス、黒点/白点とガンマの自動調整など
3. **美肌** (`retouch/skin.py`): 周囲より暗い/赤い小さな斑点を検出して周辺の肌で埋め、
   Guided Filter で凹凸をならした上に元の高周波 (肌理) を戻す周波数分離
4. **顔パーツ** (`retouch/face.py`): ランドマークから目・目の下・口のマスクを作り局所補正
5. **背景** (`retouch/background.py`): 人物マスクで前景を保護しながら背景だけをぼかす / 置き換える (境界の色にじみも除去)
6. **変形**: 小顔・デカ目は変位場を合成して最後に 1 回だけ再サンプリング (歪み・二重変形を防止)
7. **修正ブラシ** (`retouch/heal.py`): 周囲で質感が最も似た場所から肌理をコピーし、色は周囲に合わせて修復

GUI のプレビューは縮小画像で解析結果をキャッシュして高速に表示し、書き出し時はフル解像度で再解析します。
色補正は 1 本のトーンカーブ (LUT) にまとめ、顔パーツの補正は顔周辺だけ、低周波の肌補正は縮小して計算するため、
2000 万画素クラスの写真でも 1 枚あたり十数秒程度 (4 コア環境での目安) で処理できます。

## 開発

```bash
pip install -e ".[all,dev]"
pytest                      # GUI のテストは画面なし (offscreen) で動きます
```

### 配布版のビルド

```bash
pip install -e ".[all]" pyinstaller
python packaging/build.py   # dist/PicMe (macOS は dist/PicMe.app) ができる
```

`v` で始まるタグ (例: `v0.2.0`) を push すると、GitHub Actions が Windows / macOS / Linux 版をビルドして Releases に添付します。

## ライセンスについて

AI モデルは Google MediaPipe のモデル (Apache License 2.0) を初回実行時にダウンロードして使用します。
