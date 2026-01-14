import logging
import numpy as np
import keras
import tensorflow as tf
import json
from model import CarModel
from loader import Loader
from shared.models import ModelConfig

logger = logging.getLogger(__name__)


class Trainer:
    """モデル学習・変換用クラス"""

    def __init__(self):
        """Trainerを初期化"""
        pass

    def train(self) -> None:
        """
        トレーニングデータを使用してモデルを学習し、TFLiteに変換する
        
        処理フロー:
        1. ローダーからデータセットと統計情報を取得
        2. ステアリング・スロットル値を正規化
        3. CNNモデルをビルド・コンパイル
        4. 学習実行（EarlyStoppingで過学習防止）
        5. 設定情報をconfig.jsonに保存
        6. TFLiteに変換してラズパイ用に出力
        """
        # データセットと統計情報を読み込み
        loader = Loader()
        images, steerings, throttles = loader.load_sessions()
        stats = loader.get_stats()

        steering_min = stats['steering_min']
        steering_max = stats['steering_max']
        throttle_min = stats['throttle_min']
        throttle_max = stats['throttle_max']

        # ステアリング値を[-1, 1]に正規化
        steering_norm = 2 * (steerings - steering_min) / (steering_max - steering_min) - 1
        steering_norm = np.clip(steering_norm, -1, 1)

        # スロットル値を[0, 1]に正規化
        throttle_norm = (throttles - throttle_min) / (throttle_max - throttle_min)
        throttle_norm = np.clip(throttle_norm, 0, 1)
        
        # ステアリングとスロットルを結合（出力層用）
        outputs = np.column_stack([steering_norm, throttle_norm])

        # CNNモデルをビルド
        model: keras.Model = CarModel.build_model(
            input_shape=images.shape[1:],
            output_type='continuous',
        )
        model.summary()

        # モデルをコンパイル
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss='mse',
            metrics=['mae'],
        )

        # モデルを学習（EarlyStoppingで過学習を防止）
        history = model.fit(
            images,
            outputs,
            batch_size=32,
            epochs=50,
            validation_split=0.2,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True,
                ),
            ],
        )

        # 学習済みモデルを保存
        model_path = "./output/model.keras"
        model.save(str(model_path))
        logger.info(f"モデルを保存しました: {model_path}")
        
        # 設定情報をJSONに保存（ラズパイで使用）
        config = {
            'image_size': [160, 120],  # [幅, 高さ] - ラズパイ推論時の入力順序
            'image_shape': list(images.shape[1:]),  # [高さ, 幅, チャンネル] - TensorFlow内部用
            'steering_min': float(steering_min),
            'steering_max': float(steering_max),
            'throttle_min': float(throttle_min),
            'throttle_max': float(throttle_max),
            'num_samples': len(images),
            'epochs_trained': len(history.history['loss']),
            'final_loss': float(history.history['loss'][-1]),
            'final_val_loss': float(history.history['val_loss'][-1]),
        }
        config_path = "./output/config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"設定情報を保存しました: {config_path}")
        
        # TFLiteに変換
        self.convert_to_tflite()

    def convert_to_tflite(self) -> None:
        """
        KerasモデルをTensorFlow Liteに変換してラズパイ用に出力
        
        TFLiteへの変換により、ラズパイ上での推論が高速化される
        """
        # 学習済みモデルを読み込み
        model = keras.models.load_model(str("./output/model.keras"))
        logger.info("モデルをTFLiteに変換中...")
        
        # TFLite変換設定
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
        ]
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # TFLiteモデルに変換
        tflite_model = converter.convert()
        
        # TFLiteモデルを保存
        tflite_path = "./output/model.tflite"
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        logger.info(f"TFLiteモデルを保存しました: {tflite_path}")


def main() -> None:
    """
    テスト用のメインエントリーポイント
    
    Trainerクラスを初期化して学習を開始する
    """
    # ロギングの設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )
    
    logger.info("=" * 60)
    logger.info("モデル学習・変換スクリプトを開始します")
    logger.info("=" * 60)
    
    try:
        # Trainerを初期化して学習を実行
        trainer = Trainer()
        trainer.train()
        
        logger.info("=" * 60)
        logger.info("学習と変換が完了しました")
        logger.info("=" * 60)
        
    except FileNotFoundError as e:
        logger.error(f"ファイルが見つかりません: {e}")
    except Exception as e:
        logger.error(f"エラーが発生しました: {e}")
        raise

if __name__ == '__main__':
    main()
