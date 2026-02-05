import logging
import time
from pathlib import Path
import csv
import numpy as np
# import cv2
from cnn.models import DriveData
# from cnn.transformer import DataTransformer
from PIL import Image

logger = logging.getLogger(__name__)


class Recorder:
    def __init__(self, base_dir: str) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.session_id = int(time.time())
        self.session_dir = self.base_dir / f"session_{self.session_id}"
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.image_dir = self.session_dir / "image"
        self.image_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = self.session_dir / "records.csv"
        self._init_csv()

        logger.info(f"Recorder initialized: {self.session_id}")

    def _init_csv(self) -> None:
        with open(self.csv_path, 'w', newline='') as f:
            write = csv.writer(f)
            write.writerow(['timestamp', 'image', 'steering', 'throttle'])

    def save(self, frame: np.ndarray, steering: float, throttle: float) -> None:
        timestamp = time.time()
        image_name = f"{timestamp:.6f}.jpg"
        image_path = self.image_dir / image_name

        data = DriveData(timestamp=timestamp, steering=steering, throttle=throttle, image_name=image_name)

        Image.fromarray(frame).save(image_path)

        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, image_name, steering, throttle])

        logger.debug(f"saved: {data}")
