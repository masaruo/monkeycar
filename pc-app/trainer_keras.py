import logging
import numpy as np
import keras
import tensorflow as tf
import json
from pathlib import Path
from model import CarModel
from shared.models import ModelConfig

logger = logging.getLogger(__name__)


class Trainer:
    """モデル学習・変換用クラス"""

    def __init__(
        self,
        images: np.ndarray,
        steerings: np.ndarray,
        throttles: np.ndarray,
        cfg: 'ModelConfig',
    ):
        self.images = images
        self.steerings = steerings
        self.throttles = throttles
        self.cfg = cfg
        logging.info("Trainer initiated")

    def __normalize(self) -> tuple[float, float]:
        steering_min = self.cfg.steering_min
        steering_max = self.cfg.steering_max
        throttle_min = self.cfg.throttle_min
        throttle_max = self.cfg.throttle_max

        # ステアリング値を[-1, 1]に正規化 
        steering_normal = 2 * (self.steerings - steering_min) / (steering_max - steering_min) - 1
        steering_normal = np.clip(steering_normal, -1, 1)
        # スロットル値を[0, 1]に正規化 
        throttle_normal = (self.throttles - throttle_min) / (throttle_max - throttle_min)
        throttle_normal = np.clip(throttle_normal, 0, 1)

        return steering_normal, throttle_normal

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
        # 出力ディレクトリを作成
        Path('./model').mkdir(parents=True, exist_ok=True)
        Path('./output').mkdir(parents=True, exist_ok=True)
        
        steering_norm, throttle_norm = self.__normalize()
        
        # ステアリングとスロットルを結合（出力層用）
        outputs = np.column_stack([steering_norm, throttle_norm])

        # CNNモデルをビルド
        num_samples, *individual_shape = self.images.shape
        model: keras.Model = CarModel.build_model(
            input_shape=tuple(individual_shape),
            output_type='continuous',
        )
        model.summary()

        # モデルをコンパイル
        model.compile(
            # optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            optimizer='adam',
            loss='mse',
            metrics=['mae'],
        )

        # モデルを学習（EarlyStoppingで過学習を防止）
        history = model.fit(
            self.images,
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
        model_path = "./model/model.keras"
        model.save(str(model_path),)
        logger.info(f"モデルを保存しました: {model_path}")

        self.cfg.image_shape = list(self.images.shape[1:]) # [高さ, 幅, チャンネル] - TensorFlow内部用
        self.cfg.num_samples = len(self.images)
        self.cfg.epochs_trained = len(history.history['loss'])
        self.cfg.final_loss = float(history.history['loss'][-1])
        self.final_val_loss = float(history.history['val_loss'][-1])

        config_path = "./output/config.json"
        with open(config_path, 'w') as f:
            json.dump(self.cfg.model_dump(), f, indent=2)

        # TFLiteに変換
        self.__convert_to_tflite(model=model)

    def __convert_to_tflite(self, model:keras.Model) -> None:
        """
        KerasモデルをTensorFlow Liteに変換してラズパイ用に出力
        
        TFLiteへの変換により、ラズパイ上での推論が高速化される
        """
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


# def main() -> None:
#     """
#     テスト用のメインエントリーポイント
    
#     Trainerクラスを初期化して学習を開始する
#     """
#     # ロギングの設定
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     )
    
#     logger.info("=" * 60)
#     logger.info("モデル学習・変換スクリプトを開始します")
#     logger.info("=" * 60)
    
#     try:
#         # Trainerを初期化して学習を実行
#         trainer = Trainer()
#         trainer.train()
        
#         logger.info("=" * 60)
#         logger.info("学習と変換が完了しました")
#         logger.info("=" * 60)
        
#     except FileNotFoundError as e:
#         logger.error(f"ファイルが見つかりません: {e}")
#     except Exception as e:
#         logger.error(f"エラーが発生しました: {e}")
#         raise

# if __name__ == '__main__':
#     main()
