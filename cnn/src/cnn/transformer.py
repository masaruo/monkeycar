import numpy as np
import cv2
from typing import Optional
from .models import ModelConfig


class DataTransformer:
    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        image_size: tuple[int, int] = (160, 120),
    ):
        """
        Args:
            config: ModelConfig instance (required for normalization/denormalization)
            image_size: Target image size (width, height) used if config is None
        """
        self.config = config
        self.image_size = config.image_size if config else image_size

    def transform_image(self, image: np.ndarray) -> np.ndarray:
        """OpenCV形式(BGR, HWC)をモデル入力形式(RGB, CHW, 0.0-1.0)に変換。"""

        # color BGR -> RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # resize to (160, 120)
        resized = self.resize_image(rgb)

        # normalize:
        normalized = resized.astype(np.float32) / 255.0

        # 次元変換 (HWC) -> (CHW)
        transposed = normalized.transpose(2, 0, 1)

    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """Resize image to target size if necessary."""
        target_w, target_h = self.image_size
        current_h, current_w = image.shape[:2]

        if current_w == target_w and current_h == target_h:
            return image

        return cv2.resize(image, self.image_size)

    def normalize_labels(self, steering: float, throttle: float) -> tuple[float, float]:
        """物理値をモデル用に正規化

            return normalized_steering, normalized_throttle
        """
        if not self.config:
            return steering, throttle

        s_min, s_max = self.config.steering_min, self.config.steering_max
        t_min, t_max = self.config.throttle_min, self.config.throttle_max

        #ステアリングの正規化 [s_min, s_max] -> [-1.0 ~ 1.0]
        s_norm = 2 * (steering - s_min) / (s_max - s_min) - 1
        s_norm = np.clip(s_norm, -1.0, 1.0)

        #スロットルの正規化 [t_min, t_max] -> [0, 1]
        t_norm = (throttle - t_min) / (t_max - t_min)
        t_norm = np.clip(t_norm, 0.0, 1.0)

        return s_norm, t_norm

    def denormalize_labels(self, s_norm: float, t_norm: float) -> tuple[float, float]:
        """モデルの正規化出力を物理値に復元
        """
        if not self.config:
            return s_norm, t_norm

        s_min, s_max = self.config.steering_min, self.config.steering_max
        t_min, t_max = self.config.throttle_min, self.config.throttle_max

        # Steering: [-1, 1] -> [min, max]
        steering = ((s_norm + 1) / 2) * (s_max - s_min) + s_min

        # Throttle: [0, 1] -> [min, max]
        throttle = t_norm * (t_max - t_min) + t_min

        s_clip = np.clip(steering, s_min, s_max)
        t_clip = np.clip(throttle, t_max, t_min)

        return s_clip, t_clip

    def prepare_inference_input(self, image: np.ndarray) -> np.ndarray:
        #推論：(H, W, C) -> (1, C, H, W)
        transformed = self.transform_image(image=image)
        batched = np.expand_dims(transformed, axis=0)
        return (batched)

    def prepare_batch_input(self, image_list: list[np.ndarray]) -> np.ndarray:
        """
        訓練用に画像リストを (N, C, H, W) 形式のバッチに変換する。
        
        Args:
            image_list: transform_image 済みの (C, H, W) 配列のリスト
        Returns:
            (N, C, H, W) の4次元配列
        """
        if not image_list:
            return np.array([], dtype=np.float32)
        
        # リストを結合して (N, C, H, W) を作成
        batch = np.array(image_list)
        
        # バッチサイズが 1 の場合に次元が圧縮されるのを防ぎ、確実に 4 次元を保証する
        if batch.ndim == 3:
            batch = batch[np.newaxis, :, :, :]
            
        return batch.astype(np.float32)
