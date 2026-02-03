# Monkeycar

自作RCカー（Raspberry Pi）での走行・データ収集と、PCでの学習を一連で行う学習用プロジェクト。PC側で学習したモデルをラズパイへ転送し、推論・操舵を行います。学習用の自作CNNライブラリも同梱しています。

## 構成

- car-app: ラズパイ上でカメラ・ジョイスティック・モーター制御を行う走行アプリ
- pc-app: 走行データの学習パイプライン（モデル学習・転送）
- cnn: NumPyベースの自作CNNライブラリ

## 前提/環境

- Python 3.11以上
- パッケージ管理: uv
- 実行/タスク: make
- pc-app: Apple Silicon Mac で開発
- car-app: Debian bookworm（Raspberry Pi）で開発

## 基本ワークフロー

1. ホストPCとラズパイ両方に、`git clone`
1. ラズパイで走行・記録（`cd car-app`）
2. PCで学習（`cd pc-app`）
3. 学習済みモデルをラズパイへ転送（pc-app → car-app）pc-appにて`make pull / push` 
4. 推論走行（`make autopilot in car-app`）

## car-app（ラズパイ側）概要
`cd car-app`
- 起動: `make run`
- オートパイロット: `make autopilot`
- 記録モード: `make record`

### ジョイスティック操作

- X: オートパイロット ON/OFF
- B: 記録 ON/OFF
- Y: 終了（例外でループ終了）
- 左スティック X: ステアリング
- RT: スロットル

### データ/モデル

- 記録データ: `car-app/data/session_<timestamp>/`
	- 画像: `image/*.jpg`
	- ラベル: `records.csv`
- 推論モデル: `car-app/output/`（`config.json`, `params.pkl`）

## pc-app（学習側）概要
`cd pc-app`
- インストール: `make install` または `uv sync`
- 学習: `make train`
- ラズパイから取得: `make pull`
- ラズパイへ転送: `make push`

学習結果は `pc-app/weights_bin/` に出力されます（`params.pkl`, `config.json`）。

## cnn（自作CNNライブラリ）概要

- NumPyのみで実装された学習用CNN
- 形状デバッグログ対応（`LOG_LEVEL=DEBUG`）
- SGD/Adam対応

## SSH/同期メモ

### SSH

- `ssh -A team40@team40.local`
- `-A` でフォワーディングを有効にし、ラズパイからGitHubへコミット可能
- VS Code の SSH 機能を使うと便利
