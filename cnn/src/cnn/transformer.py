import numpy as np
import cv2
from typing import Optional
from .models import ModelConfig

class DataTransformer:
    def __init__(self, config: Optional[ModelConfig] = None, image_size: tuple[int, int] = (160, 120)):
        """
        Args:
            config: ModelConfig instance (required for normalization/denormalization)
            image_size: Target image size (width, height) used if config is None
        """
        self.config = config
        self.image_size = config.image_size if config else image_size

    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """Resize image to target size if necessary.
        
        Args:
            image: Input image (H, W, C)
            
        Returns:
            Resized image (H, W, C)
        """
        # image_size is (Width, Height), shape is (Height, Width, Channel)
        target_w, target_h = self.image_size
        current_h, current_w = image.shape[:2]
        
        if current_w == target_w and current_h == target_h:
            return image
            
        return cv2.resize(image, self.image_size)

    def transform_image(self, image: np.ndarray) -> np.ndarray:
        """Process image for model input (Resize -> Normalize -> CHW).

        Args:
            image: Input image (H, W, C) BGR or RGB

        Returns:
            Processed array (C, H, W)
        """
        # 1. Resize if needed
        resized = self.resize_image(image)

        # 2. Normalize [0, 255] -> [0.0, 1.0]
        normalized = resized.astype(np.float32) / 255.0

        # 3. HWC -> CHW
        transposed = normalized.transpose(2, 0, 1)

        # Note: Batch dimension is NOT added here.
        # loader.py expects (C, H, W) to append to list.
        # interpreter.py will treat this as (C, H, W) and use expand_dims to make it (1, C, H, W).
        
        return transposed

    def normalize_labels(self, steering:  float | np.ndarray, throttle:  float | np.ndarray) -> tuple[float | np.ndarray, float | np.ndarray]:
        """Normalize physical values to model range.
        Supports both single float and numpy array inputs.
        """
        if not self.config:
            raise ValueError("ModelConfig is required for normalization")

        s_min = self.config.steering_min
        s_max = self.config.steering_max
        t_min = self.config.throttle_min
        t_max = self.config.throttle_max

        # Steering: Map [min, max] -> [-1, 1]
        s_norm = 2 * (steering - s_min) / (s_max - s_min) - 1
        s_norm = np.clip(s_norm, -1.0, 1.0)

        # Throttle: Map [min, max] -> [0, 1]
        t_norm = (throttle - t_min) / (t_max - t_min)
        t_norm = np.clip(t_norm, 0.0, 1.0)

        if isinstance(steering, float):
            return float(s_norm), float(t_norm)
        return s_norm, t_norm

    # def denormalize_labels(self, steering_norm: float, throttle_norm: float) -> tuple[float, float]:
    #     """Convert model output back to physical values."""
    #     if not self.config:
    #         raise ValueError("ModelConfig is required for denormalization")

    #     s_min = self.config.steering_min
    #     s_max = self.config.steering_max
    #     t_min = self.config.throttle_min
    #     t_max = self.config.throttle_max

    #     # Steering: [-1, 1] -> [min, max]
    #     steering = ((steering_norm + 1) / 2) * (s_max - s_min) + s_min

    #     # Throttle: [0, 1] -> [min, max]
    #     throttle = throttle_norm * (t_max - t_min) + t_min

    #     return float(steering), float(throttle)
