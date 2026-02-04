import logging
import numpy as np
from pathlib import Path
from cnn.models import ModelConfig
from cnn.network import CarConvNet
from cnn.util import normalize

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


        # ConvNetworkの初期化
        # input_dim needs to be (C, H, W)
        # image_size is (W, H)
        input_dim = (3, self.image_size[1], self.image_size[0])
        self.net = CarConvNet(input_dim=input_dim)
        
        # パラメータの読み込み
        self.net.load_params(str(self.params_path))

        logger.info("DeepConvNetwork initialized")

    def predict(self, frame: np.ndarray) -> tuple[float, float]:
        #　前処理
        normalized_img = normalize(frame)
        
        # shape (h, w, c) -> (c, h, w)
        transposed = normalized_img.transpose(2, 0, 1)
        # add mock batch = 1
        batched = np.expand_dims(transposed, axis=0)
        # 推論実行
        predictions = self.net.predict(batched)
        result = predictions[0]
        steering, throttle = result
        return steering, throttle
