import logging
import time
from typing import Self
# from shared.config import ROTATE
import numpy as np
from picamera2 import Picamera2
# import cv2
from typing import Final

ROTATE: Final = 2

logger = logging.getLogger(__name__)


class Camera:
    """Picamera2でフレームを取得するクラス。"""

    def __init__(self, width: int = 320, height: int = 240, fps: int = 30) -> None:
        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(
            main={"size": (width, height), "format": "RGB888"},
            buffer_count=4,
        )
        self.picam2.configure(config)
        self.picam2.start()
        time.sleep(0.2)  # ウォームアップ
        logger.info(f"Picamera2 initialized: {width}x{height} @ {fps}fps")

    def capture(self) -> np.ndarray:
        frame = self.picam2.capture_array("main")
        if frame is None:
            raise RuntimeError("Failed to capture frame")
        frame = np.rot90(frame, ROTATE)
        
        # 切り抜く代わりに「黒く塗る」方法を使ってください↓
        frame[:120, :] = 0   # 上端10pxを黒く
        # frame[-10:, :] = 0  # 下端10pxを黒く (車体隠し含むなら -40 とか)
        # frame[:, :10] = 0   # 左端10pxを黒く
        # frame[:, -10:] = 0  # 右端10pxを黒く
        
        return frame

    def close(self) -> None:
        self.picam2.stop()
        self.picam2.close()
        logger.info("Camera closed")

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
