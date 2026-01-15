import logging
import numpy as np
import tensorflow.lite as tflite
from pathlib import Path
import json
import cv2
from shared.models import ModelConfig

logger = logging.getLogger(__name__)


class Interpreter:
    def __init__(self, base_dir: str) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.base_dir / "model.tflite"
        self.config_path = self.base_dir / "config.json"

        with open(self.config_path, 'r') as f:
            self.config = ModelConfig.model_validate_json(f.read())

        self.image_size = self.config.image_size
        self.steering_min = self.config.steering_min
        self.steering_max = self.config.steering_max
        self.throttle_min = self.config.throttle_min
        self.throttle_max = self.config.throttle_max

        self.interpreter = tflite.Interpreter(model_path=str(self.model_path))
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        logger.info(f"TFLiteモデル読み込み完了: {self.model_path}")
        logger.info(f"入力形状: {self.input_details[0]['shape']}")
        logger.info(f"出力形状: {self.output_details[0]['shape']}")

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """画像を前処理してモデル入力形式に変換

        Args:
            frame: カメラからの画像 (H, W, 3)

        Returns:
            前処理済み画像 (1, H, W, 3)
        """

        # bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # リサイズ（学習時と同じサイズに）
        resized = cv2.resize(frame, self.image_size)

        # 正規化 [0, 255] -> [0.0, 1.0]
        normalized = resized.astype(np.float32) / 255.0

        # バッチ次元を追加 (H, W, 3) -> (1, H, W, 3)
        batched = np.expand_dims(normalized, axis=0)

        return batched

    def predict(self, frame: np.ndarray) -> tuple[float, float]:
        """画像からステアリングとスロットルを予測

        Args:
            frame: カメラからの画像

        Returns:
            (steering, throttle) のタプル
            - steering: -1.0～1.0
            - throttle: -1.0~1.0
        """
        # 前処理
        input_data = self.preprocess(frame)

        # 推論実行
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        # 出力を取得
        output = self.interpreter.get_tensor(self.output_details[0]['index'])

        # 出力は (1, 2) の形状: [[steering, throttle]]
        steering = float(output[0][0])
        throttle = float(output[0][1])

        # 学習時の範囲にクリップ
        steering = np.clip(steering, self.steering_min, self.steering_max)
        throttle = np.clip(throttle, self.throttle_min, self.throttle_max)

        return steering, throttle
