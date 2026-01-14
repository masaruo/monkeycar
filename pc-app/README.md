    image_size: tuple[int, int] = Field(..., description="[Width, Height]")
    image_shape: List[int] = Field(description="[Height,  Width, Channels]")
    
    # 制御値の正規化用
    steering_min: float = Field(..., ge=-1.0, le=1.0)
    steering_max: float = Field(..., ge=-1.0, le=1.0)
    throttle_min: float = Field(..., ge=-1.0, le=1.0)
    throttle_max: float = Field(..., ge=-1.0, le=1.0)
    normalize_param: float
    # 訓練メタデータ (Optional)
    num_samples: int
    epochs_trained: int
    final_loss: float
    final_val_loss: float

# PC用訓練パイプライン

ホストPC上で行う機械学習訓練スクリプトです。Donkeycar方式と同様です。

## セットアップ

```bash
# パッケージのインストール
make sync-pc

# または開発モード（Jupyter含む）
make sync-dev
```

## 訓練パイプライン

### 1️⃣  データセット確認

```bash
cd src/pc
python -c "from data_loader import DataLoader; loader = DataLoader('../../../data'); images, s, t = loader.load_sessions(); print(loader.get_stats())"
```

### 2️⃣  データセット可視化

```bash
cd src/pc
python visualize.py
```

実行後、以下が生成されます:
- `dataset_samples.png` - サンプル画像
- `dataset_statistics.png` - ステアリング/スロットル分布

### 3️⃣  モデル訓練

```bash
cd src/pc
python model_train.py
```

出力:
- `models/model.h5` - Kerasモデル
- `models/config.json` - 訓練設定（正規化パラメータ含む）

#### 訓練パラメータをカスタマイズする場合:

[model_train.py](model_train.py) の最後を編集:

```python
trainer = Trainer(
    data_dir='../../../data',
    image_size=(160, 120),      # 画像サイズ
    batch_size=32,              # バッチサイズ
    epochs=100,                 # エポック数
    val_split=0.2,              # 検証データの割合
    output_dir='./models',
)
```

### 4️⃣  TFLite変換（ラズパイ用）

```bash
cd src/pc
python model_convert.py
```

出力:
- `models/model.tflite` - ラズパイで実行可能なモデル
- `models/config.json` - 設定ファイル（推論時に必要）

### 5️⃣  ラズパイに転送

```bash
scp src/pc/models/model.tflite pi@raspberrypi.local:~/minicar-car/src/car/models/
scp src/pc/models/config.json pi@raspberrypi.local:~/minicar-car/src/car/models/
```

---

## スクリプト説明

### `data_loader.py`

複数のセッションからデータを読み込みます:

```python
from data_loader import DataLoader

loader = DataLoader('../../../data', image_size=(160, 120))
images, steerings, throttles = loader.load_sessions()
```

**特徴:**
- 複数セッション自動読み込み
- 画像の自動リサイズ・正規化
- 統計情報の計算

### `model_train.py`

CNNモデルを訓練します:

```python
from model_train import Trainer

trainer = Trainer(data_dir='../../../data')
model, config = trainer.run()
```

**モデル構成:**
- 入力: 160×120 RGB画像
- 3層CNN + 全結合層
- 出力: [ステアリング, スロットル]
- Early Stopping で過学習防止

**出力の正規化:**
- ステアリング: [-1, 1]
- スロットル: [0, 1]

### `model_convert.py`

KerasモデルをTFLiteに変換します:

```bash
python model_convert.py
```

**変換オプション:**
- `quantize=True`: 量子化で高速化・省メモリ

### `visualize.py`

データセット統計を可視化:

```bash
python visualize.py
```

---

## トラブルシューティング

### ❌ "No module named 'tensorflow'"

```bash
make sync-pc  # パッケージをインストール
```

### ❌ データが読み込めない

- `records.csv` と `image/` が同じディレクトリにあるか確認
- 画像ファイル名が CSV の `image` 列と一致しているか確認

```bash
# データ構造確認
ls -la data/session_1768186952/
ls -la data/session_1768186952/image/ | head
```

### ❌ メモリ不足エラー

バッチサイズを減らす:

```python
trainer = Trainer(batch_size=16)  # 32 → 16
```

### ❌ TFLiteモデルがラズパイで動かない

`config.json` も一緒に転送しているか確認:

```bash
scp src/pc/models/config.json pi@raspberrypi.local:~/minicar-car/src/car/models/
```

---

## 次のステップ

1. ラズパイで [minicar-car](https://github.com/your-repo/minicar-car) の `inference.py` を実装
2. 推論スクリプトで `model.tflite` と `config.json` を読み込む
3. `main.py` で自動走行を実行

