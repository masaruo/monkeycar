import logging
import numpy as np
from pathlib import Path
import cv2
from cnn.transformer import DataTransformer
from cnn.models import ModelConfig
from cnn.network import CarConvNet

logger = logging.getLogger(__name__)


class Inference:
    def __init__(self, base_dir: str) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.params_path = self.base_dir / "params.pkl" # パラメータファイルのパス
        self.config_path = self.base_dir / "config.json"

        with open(self.config_path, 'r') as f:
            self.config = ModelConfig.model_validate_json(f.read())

        self.image_size = self.config.image_size
        self.steering_min = self.config.steering_min
        self.steering_max = self.config.steering_max
        self.throttle_min = self.config.throttle_min
        self.throttle_max = self.config.throttle_max

        self.transformer = DataTransformer(config=self.config)

        # ConvNetworkの初期化
        # input_dim needs to be (C, H, W)
        # image_size is (W, H)
        input_dim = (3, self.image_size[1], self.image_size[0])
        self.net = CarConvNet(input_dim=input_dim)
        
        # パラメータの読み込み
        self.net.load_params(str(self.params_path))

        logger.info("DeepConvNetwork initialized")

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """画像を前処理してモデル入力形式に変換

        Args:
            frame: カメラからの画像 (H, W, 3)

        Returns:
            前処理済み画像 (1, 3, H, W)  <-- NCHW形式に変更
        """

        # Transformerによる一括処理
        # (Resize -> Normalize -> CHW)
        # transposed = self.transformer.transform_image(frame)

        # バッチ次元を追加 (3, H, W) -> (1, 3, H, W)
        # batched = np.expand_dims(transposed, axis=0)

        return batched

    def predict(self, frame: np.ndarray) -> tuple[float, float]:
        """画像からステアリングとスロットルを予測

        Args:
            frame: カメラからの画像

        Returns:
            (steering, throttle) のタプル
            - steering: -1.0～1.0 (Physical value depends on config)
            - throttle: -1.0~1.0 (Physical value depends on config)
        """
        # 前処理
        input_data = self.preprocess(frame)

        # 推論実行
        output = self.net.predict(input_data)

        # 出力を取得 (1, 2) の形状: [[steering_norm, throttle_norm]]
        steering_norm = float(output[0][0])
        throttle_norm = float(output[0][1])

        # 非正規化 (モデル出力[-1, 1] -> 物理値[min, max])
        steering, throttle = self.transformer.denormalize_labels(steering_norm, throttle_norm)

        return steering, throttle
