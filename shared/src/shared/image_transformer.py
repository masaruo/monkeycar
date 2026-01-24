import numpy as np
import logging
import cv2
from shared.config import H, W

logger = logging.getLogger(__name__)


class ImageTransformer:
    def __init__(self) -> None:
        pass

    def resize_image(self, frame: np.ndarray) -> np.ndarray:
        return cv2.resize(frame, (H, W))

    def transform_image(self, frame: np.ndarray) -> np.ndarray:
        resized = self.resize_image(frame)
        # [0 ~ 255] -> [0.0 ~ 1.0]
        normalized = resized.astype(np.float32) / 255.0

        # [HWC] -> [CHW]
        transposed = normalized.transpose(2, 0, 1)

        return transposed

    def normalize(self) -> set[np.ndarray, np.ndarray]:
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

    def denormalize(self) -> set[float, float]:

    def inference_image(self, frame: np.ndarray) -> np.ndarray:
        transposed = self.transform_image(frame)

        # add 1dim for batch
        batched = np.expand_dims(transposed, axis=0)

        return batched