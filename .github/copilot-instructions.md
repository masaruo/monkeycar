# Minicar Project - Copilot Instructions

## プロジェクト概要

このプロジェクトは、Donkeycar等の既存フレームワークを使わずに、自律走行の基本要素（データ収集・学習・推論・車両制御）を最小構成で学ぶための学習用プロジェクトです。

本プロジェクトは用途別に3つのアプリケーションへ分割されています。

- `pc-app`: ホストPCでの機械学習・データ処理
- `car-app`: Raspberry Piでのハードウェア制御・推論実行
- `shared`: 共有ライブラリ（型定義・共通ユーティリティ）

## プロジェクト構成

```
car-app/                 # ラズパイ用（ハードウェア制御・推論）
  camera.py              # カメラ画像取得
  config.py              # 推論・制御向け設定
  interpreter.py         # TFLite推論ラッパー
  joystick.py            # ジョイスティック入力
  main.py                # メインループ（自律走行）
  motor.py               # PCA9685でのサーボ・ESC制御
  recorder.py            # 走行ログ・データ収集（必要に応じて）
  Makefile
  pyproject.toml

pc-app/                  # ホストPC用（学習・前処理）
  loader.py              # データセット読み込み・前処理
  main.py                # 実行エントリ（例: 学習/評価）
  model.py               # モデル定義（TensorFlow/Keras）
  trainer.py             # モデル学習スクリプト
  test_tf.py             # TF動作テスト（環境確認用）
  data/                  # 収集データ（画像・CSV）
  output/                # 学習成果物（model.keras, model.tflite, config.json）
  sample/                # 参考サンプル（data_loader.pyなど）
  Makefile
  pyproject.toml

shared/                  # 共有Pythonパッケージ
  src/shared/__init__.py
  src/shared/py.typed    # 型情報マーカー
  pyproject.toml
  README.md
```

## ハードウェア構成

- シャーシ: タミヤ TT-02（1/10 RCカー）
- コンピュータ: Raspberry Pi 4 Model B (8GB)
- PWMドライバー: PCA9685 (16ch/12-bit, I2C: 0x40)
- ESC: タミヤ TEU-107BK（連続最大電流 前後進とも75A）
- サーボ: タミヤ ファインスペック2.4G システム付属
- カメラ: サインスマート 広角魚眼レンズカメラ (Raspberry Pi用)
- RCシステム: タミヤ ファインスペック2.4G 電動RCドライブセット
- ジョイスティック: Logicool F310 ゲームパッド

## 技術スタック

- 言語: Python 3.9～3.13（TensorFlow互換性考慮）
- 学習（`pc-app`）: TensorFlow/Keras, OpenCV
- 推論（`car-app`）: TensorFlow Lite, picamera2, adafruit-circuitpython-servokit, RPi.GPIO
- 共有（`shared`）: 型定義・共通ユーティリティ
- パッケージ管理: uv（`pip`ではなく`uv pip`を使用）

## 開発環境のセットアップ

### ホストPC（macOS/Linux/Windows） - pc-app

初回セットアップ:
```bash
# uvのインストール
curl -LsSf https://astral.sh/uv/install.sh | sh

# リポジトリをクローン
git clone <repo-url> monkeycar
cd monkeycar

# pc-appの依存関係をインストール
cd pc-app
make sync
```

開発フロー:
```bash
# VS Codeでpc-appを開く
code pc-app

# 学習・前処理（例）
python trainer.py        # モデル学習
python main.py           # 実行エントリ（評価/可視化など）
```

### ラズパイ - car-app（VS Code Remote SSH推奨）

初回セットアップ（ラズパイ上）:
```bash
# uvのインストール
curl -LsSf https://astral.sh/uv/install.sh | sh

# リポジトリをクローン（例: /home/pi/monkeycar）
git clone <repo-url> monkeycar
cd monkeycar/car-app

# car-appの依存関係をインストール
make sync
```

開発フロー（VS Code Remote SSH）:
```bash
# ホストPCのVS Codeから
1. Cmd/Ctrl+Shift+P → "Remote-SSH: Connect to Host"
2. ラズパイに接続（例: pi@raspberrypi.local）
3. /home/pi/monkeycar/car-app を開く
4. ファイル（camera.py, motor.py, main.pyなど）を編集
5. ラズパイのターミナルで動作テスト
```

補足:
- ラズパイ専用パッケージ（RPi.GPIO、picamera2等）はmacOS/Windowsではインストール不可
- Remote SSHによりラズパイ上のソースをVS Codeで編集でき、補完も利用可能

## 開発フロー概要（pc-app → car-app）

1. 学習（pc-app）
   ```bash
   cd pc-app
   python trainer.py        # 学習して output/ に成果物を保存
   ```

2. 変換（pc-app, 必要に応じて）
   - すでに `output/model.tflite` が生成される構成の場合は省略
   - サンプルに沿って変換する場合:
   ```bash
   python sample/model_convert.py
   ```

3. モデル転送（pc-app → car-app）
   ```bash
   # 例: tfliteモデルをcar-appへコピー
   scp pc-app/output/model.tflite pi@raspberrypi.local:/home/pi/monkeycar/car-app/model.tflite
   ```
   - 実際の読み込みパスは `car-app/config.py` や `interpreter.py` の設定に合わせて調整してください。

4. 推論実行（car-app）
   ```bash
   ssh pi@raspberrypi.local
   cd ~/monkeycar/car-app
   python main.py
   ```

## コーディング規約

基本方針:
- シンプルで読みやすいコードを優先
- 教育目的のため、過度な抽象化は避ける

スタイル:
- PEP8準拠
- コメントは必ず日本語で記述
- 型ヒント（type hints）推奨

命名規則:
- クラス名: PascalCase
- 関数・変数名: snake_case
- 定数: UPPER_SNAKE_CASE
- プライベート: _leading_underscore

## 実装する主要機能

1. カメラ制御（`car-app/camera.py`）
- picamera2で画像取得
- 前処理（リサイズ、正規化）
- 収集用途での保存（必要に応じて `recorder.py`）

2. モーター・サーボ制御（`car-app/motor.py`）
- PCA9685でPWM制御
- ステアリング（サーボ）
- スロットル（ESC）
- キャリブレーション

3. データ収集（`car-app/recorder.py` と `pc-app/loader.py`）
- ジョイスティック運転の記録（画像＋ステアリング/スロットル）
- タイムスタンプ管理、CSV/JSONメタデータ

4. 機械学習（`pc-app/model.py` と `pc-app/trainer.py`）
- 軽量CNN（MobileNet/EfficientNetなど）
- 入力: カメラ画像、出力: ステアリング/スロットル（回帰または分類）
- TFLite変換（`output/model.tflite`）

5. 自律走行（`car-app/main.py` と `car-app/interpreter.py`）
- 推論結果に基づくリアルタイム制御ループ
- セーフティ（緊急停止、フェイルセーフ）

## 重要な実装ポイント

PCA9685の制御:
- I2Cアドレス: 0x40（デフォルト）
- PWM周波数: 50Hz（RCサーボ標準）
- パルス幅: 1000～2000μs（1～2ms）
- 中央値のキャリブレーションが重要

カメラ画像:
- 解像度: 160x120 または 320x240（速度とのバランス）
- 色空間: RGB（学習時）
- 前処理: 正規化（0-1またはz-score）
- 魚眼レンズの歪み補正も検討

データ形式:
- 画像: JPEG/PNG
- メタデータ: JSON/CSV
- ディレクトリ: セッションのタイムスタンプベース

モデル設計:
- 軽量CNN推奨、過学習に注意
- 入力サイズはカメラ解像度に合わせる

## 開発ワークフロー（詳細）

1. ハードウェアテスト（car-app）: 各コンポーネントの動作確認
2. データ収集（car-app）→ データ整理（pc-app）
3. モデル学習（pc-app）→ 出力を `pc-app/output/` に保存
4. モデル変換（必要時）→ TFLite
5. モデル転送（pc→car）→ 推論速度確認（car-app）
6. 自律走行テスト（安全な環境で）

## 注意事項

安全性:
- 緊急停止機能を必ず実装
- バッテリー電圧の監視
- センサー異常時の動作定義
- テストは安全な場所で実施

パフォーマンス:
- ラズパイの処理能力に留意（カメラ＋推論）
- 目標フレームレート: 10fps以上
- 目標推論時間: 100ms以内

トラブルシューティング:
- I2C認識: `i2cdetect -y 1`
- カメラ確認: `vcgencmd get_camera`
- PWM不調: パルス幅計算と周波数の確認
- TFLite不調: ランタイムバージョンの確認

## 参考と運用ルール

- コメントは日本語で記述
- 適切なエラーハンドリングとログ出力（日本語メッセージ）
- ハードウェア制御は安全性最優先
- 主な開発は `car-app/` と `pc-app/` 配下で実施、共通処理は `shared/` へ

## パッケージ管理（uv）

- 依存は各 `pyproject.toml` に記載
- インストールは `uv pip install` を使用
- 例（pc-app）:
```bash
cd pc-app
uv pip install -e .
```
- 例（car-app）:
```bash
cd car-app
uv pip install -e .
```
