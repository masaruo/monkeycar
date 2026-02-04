import logging
import time
from typing import Self
import numpy as np
from picamera2 import Picamera2
from typing import Final


ROTATE: Final = 2

logger = logging.getLogger(__name__)


class Camera:
    """Picamera2でフレームを取得するクラス。"""

    def __init__(self, width: int=160, height: int=120, fps: int = 30) -> None:
        self.picam2 = Picamera2()
        config = self.picam2.create_video_configuration(
            main={
                "size": (width, height), 
                "format": "RGB888"},
        )
        config["main"]["framerate"] = fps
        self.picam2.configure(config)
        self.picam2.start()

        time.sleep(0.2)  # ウォームアップ
        logger.info(f"Picamera2 initialized: {width}x{height} @ {fps}fps")

    def capture(self) -> np.ndarray:
        frame = self.picam2.capture_array("main")
        if frame is None:
            raise RuntimeError("Failed to capture frame")
        frame = np.rot90(frame, ROTATE) #カメラの設置が逆さま
        # frame[:120, :] = 0 #todo 上端のマスク

        return frame

    def close(self) -> None:
        self.picam2.stop()
        self.picam2.close()
        logger.info("Camera closed")

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
