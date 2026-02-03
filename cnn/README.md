# CNN - 自作CNNライブラリ

NumPyベースで実装した、学習用の畳み込みニューラルネットワークライブラリです。自動運転車（Monkeycar）のステアリング制御など、画像からの回帰タスクに使用できます。

## 📋 目次

- [特徴](#特徴)
- [環境構築](#環境構築)
- [プロジェクト構造](#プロジェクト構造)
- [使い方](#使い方)
- [デバッグ機能](#デバッグ機能)
- [開発](#開発)

## ✨ 特徴

- **純粋なNumPy実装** - 学習目的で、内部実装を理解しやすい
- **形状デバッグ機能** - 各レイヤーの入出力形状を自動ログ出力
- **モジュラー設計** - レイヤーを組み合わせてネットワークを構築
- **パラメータ管理** - 学習可能なパラメータの自動追跡と保存
- **複数のオプティマイザー** - SGD, Adam対応

## 🚀 環境構築

### 必要なもの

- Python 3.11以上
- [uv](https://github.com/astral-sh/uv) (推奨パッケージマネージャー)

### インストール

```bash
# リポジトリをクローン
cd /path/to/cnn

# 依存関係をインストール
make install
# または
uv sync
```

## 📁 プロジェクト構造

```
cnn/
├── Makefile              # 開発タスクの自動化
├── pyproject.toml        # プロジェクト設定・依存関係
├── README.md             # このファイル
└── src/
    └── cnn/
        ├── __init__.py
        ├── layers.py      # レイヤー実装 (Convolution, Affine, Relu, etc.)
        ├── network.py     # ネットワーク構造 (CarConvNet)
        ├── optimizer.py   # オプティマイザー (SGD, Adam)
        ├── parameter.py   # パラメータ管理
        ├── util.py        # ユーティリティ (im2col, col2im)
        ├── functions.py   # 活性化関数など
        ├── models.py      # データモデル
        └── transformer.py # データ変換
```

## 🎯 使い方

### 基本的な使用例

```python
import numpy as np
from cnn.network import CarConvNet
from cnn.optimizer import Adam

# ネットワークの作成
network = CarConvNet(
    input_dim=(3, 120, 160),  # (channels, height, width)
    hidden_size=100,
    output_size=2             # steering, throttle
)

# オプティマイザーの設定
optimizer = Adam(lr=0.001)

# 学習ループ
for epoch in range(num_epochs):
    for batch_x, batch_y in train_loader:
        # 順伝播
        loss = network.loss(batch_x, batch_y)
        
        # 逆伝播
        network.gradient(batch_x, batch_y)
        
        # パラメータ更新
        optimizer.update(network.params())

# モデルの保存
network.save_params('model.pkl')

# モデルの読み込み
network.load_params('model.pkl')
```

### ネットワーク構造

`CarConvNet` は以下の構成で実装されています：

```
Input (3, 120, 160)
  ↓
Conv(16) → Relu → Pool
  ↓
Conv(32) → Relu → Pool
  ↓
Conv(64) → Relu → Pool
  ↓
Flatten
  ↓
Affine(100) → Relu → Dropout(0.5)
  ↓
Affine(2)
  ↓
Output (steering, throttle)
```

### 利用可能なレイヤー

- **Convolution**: 畳み込み層
- **Affine**: 全結合層
- **Relu**: ReLU活性化関数
- **Pooling**: Max Pooling
- **Flatten**: 多次元配列を1次元に変換
- **Dropout**: ドロップアウト（過学習防止）
- **MeanSquaredError**: 損失関数（回帰用）

### オプティマイザー

- **SGD**: 確率的勾配降下法
- **Adam**: Adaptive Moment Estimation

## 🐛 デバッグ機能

### 形状ログの有効化

環境変数 `LOG_LEVEL=DEBUG` で、各レイヤーの入出力形状を確認できます：

```bash
LOG_LEVEL=DEBUG python your_script.py
```

**出力例:**
```
[Convolution] forward IN[(32, 3, 120, 160)]:W[(16, 3, 3, 3)]:b[(16,)]:OUT[(32, 16, 60, 80)]
[Relu] forward IN[(32, 16, 60, 80)] | OUT[(32, 16, 60, 80)]
[Pooling] forward IN[(32, 16, 60, 80)] | OUT[(32, 16, 30, 40)]
[Affine] forward IN[(32, 256)]:W[(256, 100)]:b[(100,)]:OUT[(32, 100)]
```

### トレーニングスクリプトでの使用

```python
import logging
import os

# 環境変数からログレベルを取得
log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s %(levelname)s %(message)s"
)


## 🛠️ 開発

### Makefileコマンド

```bash
# すべてセットアップ（依存関係インストール）
make
# または
make install

# リント＆型チェック（緩い）
make lint

# リント＆型チェック（厳密）
make lint-strict

# キャッシュクリーンアップ
make clean

# 完全クリーンアップ（仮想環境削除）
make fclean
```

### リント・型チェック

このプロジェクトは以下のツールを使用しています：

- **Ruff**: 高速なPythonリンター
- **Mypy**: 静的型チェッカー

```bash
# 通常のチェック
make lint

# 厳密なチェック（--strict）
make lint-strict
```

### 開発ワークフロー

1. **環境セットアップ**
   ```bash
   make install
   ```

2. **コード編集**
   - `src/cnn/` 内のファイルを編集

3. **型チェック・リント**
   ```bash
   make lint
   ```

4. **動作確認**
   ```bash
   LOG_LEVEL=DEBUG python your_script.py
   ```

## 📚 学習リソース

このライブラリは学習目的で作成されています。以下の概念を理解するのに役立ちます：

- **畳み込み演算**: `im2col` を使った効率的な実装
- **逆伝播**: 各レイヤーでの勾配計算
- **パラメータ更新**: オプティマイザーの動作
- **CNNの構造**: レイヤーの組み合わせ方

### 重要な実装ポイント

1. **im2col/col2im**: 畳み込みを行列積に変換する技術
2. **Parameter クラス**: 学習可能なパラメータの自動追跡
3. **デコレーター `@log_shapes`**: メソッド実行時の形状ログ

## 📝 ライセンス

このプロジェクトは学習・研究目的で作成されています。

---

**Note**: このライブラリは教育目的で作成されており、本番環境での使用は想定していません。高速化が必要な場合は、PyTorch や TensorFlow などのフレームワークを使用してください。
