
# monkeycar car-app

ラズパイ上でカメラ・ジョイスティック・モーター制御を行う走行アプリ。**Docker** で実行します。

## 要件
- ラズパイ環境（PiCamera2 / ServoKit など）
- Debian Bookworm ベース（Raspberry Pi OS 推奨）
- Docker & Docker Compose

## ラズパイ初期設定

**初回のみ実施**（既に設定済みの場合はスキップ可能）

### 1. OS インストール
- **Raspberry Pi Imager** でラズパイに Raspberry Pi OS (other) -> Raspberry Pi OS (Legacy, 64-bit) `A port of Debian Bookworm with security updates and desktop environment` 1.2GB をインストール
- ホスト名、ユーザー情報、SSH 設定などを事前に入力

### 2. Docker Engine インストール
```bash
# 公式スクリプトでインストール（推奨）
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# またはそれぞれをインストール
sudo apt update
sudo apt install -y docker.io docker-compose containerd
```

### 3. Docker ユーザーグループ設定（sudo なし実行）
```bash
sudo usermod -aG docker $USER
# 反映させるため再ログインが必要
newgrp docker
```

### 4. I2C & カメラの有効化
```bash
sudo raspi-config
```
以下を有効に：
- **Interface Options** → **I2C** → Enable
- **Interface Options** → **Camera** → Enable
- **System Options** → **Boot** → Desktop / CLI から選択
- 再起動後、確認：
  ```bash
  ls /dev/i2c-1  # I2C デバイス確認
  ls /dev/video0 # カメラ確認
  ```

### 5. システムアップデート（推奨）
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y git vim curl
git clone git@github.com:masaruo/monkeycar.git
```

## セットアップ

### 1. ホストOS初期設定
```bash
cd ./monkeycar/car-app/
chmod +x ./setup.sh
./setup.sh
```
以下を実施：
- `python3-libcamera` のインストール
- ラズパイの `/boot/firmware/config.txt` に以下が有効になっているか確認：
  - `camera_auto_detect=1`
  - `dtparam=i2c_arm=on`

### 2. Docker イメージのビルド
```bash
make install
# または
make build
```
初回のみ実行。イメージメタデータと依存関係を準備します。

## 実行

### 通常実行
```bash
make run
```
コンテナを起動し、メインアプリを実行します。ログをリアルタイムで表示。終了は `Ctrl+C`。

### オプション実行
```bash
make autopilot   # 推論モードで起動
make record      # 記録モードで起動
```

### コンテナ内シェル
```bash
make shell
```
デバッグやファイル確認用。コンテナ内で `bash` を実行。

## 操作方法（ジョイスティック）
- X: オートパイロット ON/OFF
- B: 記録 ON/OFF
- Y: 終了（例外でループ終了）
- 左スティック X 軸: ステアリング
- RT: スロットル

## Docker 環境

### Dockerfile
- ベース: `python:3.11-slim-bookworm`（ラズパイOS と同じ Bookworm）
- `libcamera` + SDL2 依存ライブラリをインストール
- `uv` で Python 依存を管理実行時は `uv run python main.py` をエントリーポイント

### docker-compose.yaml
- **privileged モード** : GPIO / I2C / カメラへの全アクセス権限を許可
- **host ネットワーク** : ホストと同一ネットワーク（ジョイスティックなど）
- **ボリュームマウント** :
  - `.:/app` : ソースコード（変更即反映）
  - `./data:/app/data` : 記録データの永続化
  - `./output:/app/output` : 推論結果の永続化
- **デバイス指定** :
  - `/dev/video0` : カメラ
  - `/dev/i2c-1` : サーボ制御（ServoKit）
  - `/dev/input` : ジョイスティック

## 設定ファイル

### Dockerfile を詳しく
ハードウェア（libcamera, GPIO, I2C）へのアクセスが必要なため、以下を組み込み：
- `python3-libcamera` : ラズパイのカメラライブラリ
- 必要なシステムライブラリ（SDL2 等）

### setup.sh を詳しく
ホスト側で1回実施：
```bash
#!/bin/bash
# ホストOS側で必要な設定
sudo apt update && sudo apt install -y python3-libcamera

# I2Cとカメラの有効化確認（config.txtへの反映は手動またはsedで）
echo "Please ensure 'camera_auto_detect=1' and 'dtparam=i2c_arm=on' are in /boot/firmware/config.txt"
```

## データ/モデル
- 記録データ: `./data/session_<timestamp>/`
	- 画像: `image/*.jpg`
	- ラベル: `records.csv`
- 推論モデル: `./output/`
	- `config.json`
	- `params.pkl`

## Make ターゲット

### 実行・管理コマンド
- `make run` : メインアプリ実行（通常モード）
- `make autopilot` : 推論モード起動
- `make record` : 記録モード起動
- `make calibrate` : モーターESCキャリブレーション
- `make shell` : コンテナ内シェルを起動（デバッグ用）

### ビルド・管理
- `make install` / `make build` : Dockerイメージをビルド（初回必須）
- `make down` : コンテナ停止・削除

### ローカル開発向け
- `make clean` : `__pycache__` を削除
- `make lint` : ruff + mypy（通常）
- `make lint-strict` : mypy strict チェック
- `make clean-data` : `./data` と `./output` を削除
- `make fclean` : `clean` + イメージ削除

## 共有/運用メモ
### Shared
- shared がアップデートされたら `git pull` してラズパイ上にも反映する

### SSH
- `ssh -A team40@team40.local`（`-A` でフォワーディング）
- VS Code の SSH 機能を使うと便利

### rsync
- `rsync -rv [src] [dst]`
- 例: `rsync -rv data team40@team40.local:/home/team40/monkeycar/data`
