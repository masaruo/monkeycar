# Monkeycar PC Training App

ラズパイで収集したデータをPCで学習し、学習済みモデルをラズパイに転送するための訓練パイプラインです。

## 📋 必要環境

- Python 3.11以上
- [uv](https://github.com/astral-sh/uv) (Pythonパッケージマネージャー)

## 🚀 セットアップ

### 1. 依存関係のインストール

```bash
make install
# または
uv sync
```

## 📂 プロジェクト構造

```
pc-app/
├── trainer.py           # 学習スクリプト
├── loader.py            # データローダー
├── Makefile            # タスクランナー
├── sync.sh             # ラズパイとのデータ同期スクリプト
├── pyproject.toml      # プロジェクト設定
├── data/               # 学習データ（ラズパイから取得）
│   ├── session_shortcut_train/
│   └── session_shortcut_test/
└── weights_bin/        # 訓練済みモデルの出力先
    ├── params.pkl      # ネットワークの重み
    └── config.json     # モデル設定
```

## 🔄 ワークフロー

### 1️⃣ ラズパイからデータを取得

```bash
make pull
# または
./sync.sh pull
```

ラズパイ (`team40@team40.local`) から学習データを `./data` にダウンロードします。

### 2️⃣ モデルを訓練

```bash
make train
# または
make
```

デフォルトで以下の設定で訓練を実行:
- オプティマイザー: `SGD`
- 学習率: `0.01`
- バッチサイズ: `32`
- エポック数: `150`

#### デバッグモードで実行

```bash
LOG_LEVEL=debug make train
```

### 3️⃣ 訓練済みモデルをラズパイに転送

```bash
make push
# または
./sync.sh push
```

`./weights_bin` 内のモデルファイルをラズパイに転送します。

## ⚙️ カスタマイズ

### オプティマイザーの変更

`trainer.py` の最後の部分を編集:

```python
# Adam を使う場合
trainer = Trainer(
    data_dir="./data",
    batch_size=32,
    learning_rate=0.0001,
    optimizer=Adam
)

# SGD を使う場合
trainer = Trainer(
    data_dir="./data",
    batch_size=32,
    learning_rate=0.01,
    optimizer=SGD
)
```

### 訓練データセットの変更

`trainer.py` の `TRAIN_SESSIONS` と `TEST_SESSIONS` を編集:

```python
TRAIN_SESSIONS: Final = [
    'session_shortcut_train',
    # 'session_YYYYMMDD',  # 追加のセッション
]

TEST_SESSIONS: Final = [
    'session_shortcut_test',
]
```

## 🧹 クリーンアップ

### キャッシュをクリア

```bash
make clean
```

`__pycache__`, `.mypy_cache`, `.ruff_cache` などを削除します。

### 完全クリーンアップ（仮想環境も削除）

```bash
make fclean
```

`.venv` と `uv.lock` も削除されます。

### データとモデルを削除

```bash
make clean-data
```

`./data`, `./weights_bin` などを削除します。

## 🧪 コード品質チェック

### 通常のリント

```bash
make lint
```

`ruff` と `mypy` でコードをチェックします。

### 厳格なリント

```bash
make lint-strict
```

より厳格な型チェックを実行します。

## 📊 出力ファイル

訓練後、`weights_bin/` に以下が生成されます:

- **params.pkl** - ネットワークの重みパラメータ
- **config.json** - モデル設定（画像サイズ、ステアリング/スロットル範囲、損失値など）

```json
{
  "image_size": [160, 120],
  "steering_min": -1.0,
  "steering_max": 1.0,
  "throttle_min": 0.0,
  "throttle_max": 0.5,
  "num_samples": 1234,
  "final_loss": 0.2039,
  "final_val_loss": 0.1763
}
```

## 🐛 トラブルシューティング

### ラズパイに接続できない

`sync.sh` の接続設定を確認:
```bash
PI_USER=team40
PI_HOST="team40.local"
```

### メモリ不足エラー

バッチサイズを小さくする:
```python
trainer = Trainer(batch_size=16, ...)  # 32 → 16
```

### データが見つからない

`data/` ディレクトリに正しいセッションフォルダがあるか確認:
```bash
ls -la data/
```

## 📝 Makeコマンド一覧

| コマンド | 説明 |
|---------|------|
| `make` または `make train` | モデルを訓練 |
| `make install` | 依存関係をインストール |
| `make pull` | ラズパイからデータを取得 |
| `make push` | モデルをラズパイに転送 |
| `make clean` | キャッシュを削除 |
| `make fclean` | 完全クリーンアップ |
| `make clean-data` | データとモデルを削除 |
| `make lint` | コード品質チェック |
| `make lint-strict` | 厳格な型チェック |

## 🔗 関連プロジェクト

- `../cnn` - CNNモデルとオプティマイザーの実装
- ラズパイ側アプリ (`team40.local:/home/team40/monkeycar/car-app/`)
