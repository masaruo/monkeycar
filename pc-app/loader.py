import logging
import math
from typing import NamedTuple, List, Optional, Generator, Self
import random
from pathlib import Path
import numpy as np
import cv2
import pandas as pd

logger = logging.getLogger(__name__)


class Batch(NamedTuple):
    images: np.ndarray # (batch, C, H, W)
    labels: np.ndarray # batch:2 [steering, throttle]


class BatchLoader:
    def __init__(
        self,
        data_dir: str | Path = "./data",
        target_sessions: Optional[List[str]] = None,
        batch_size: int = 32,
        shuffle: bool = True
        ) -> None:

        self.data_dir: Path = Path(data_dir).resolve()
        self.target_sessions = target_sessions
        self.batch_size = batch_size
        self.shuffle: bool = shuffle

        self.samples: list = []

        self._scan_sessions()

        self.num_samples = len(self.samples)
        self.steps_per_epoch = math.ceil(self.num_samples / self.batch_size) if self.num_samples > 0 else 0

        logger.info(f"Loader initialized. Sessions: {len(target_sessions) if target_sessions else 'ALL'}, Samples: {self.num_samples}")

    def _scan_sessions(self) -> None:
        all_sessions = sorted([
            d for d in self.data_dir.iterdir()
            if d.is_dir() and d.name.startswith("session_")
        ])

        if self.target_sessions:
            target_set = set(self.target_sessions)
            sessions_dir = [d for d in all_sessions if d.name in target_set]
        else:
            sessions_dir = all_sessions

        if not sessions_dir:
            logger.warning("No valid session directories found.")
            return

        for d in sessions_dir:
            csv_path = d / 'records.csv'
            if not csv_path.exists():
                logger.debug(f"Skipping {d.name}: records csv not found.")
                continue
            try:
                df = pd.read_csv(csv_path)
                image_dir = d / 'image'

                for _, row in df.iterrows():
                    self.samples.append({
                        'path': str(image_dir / row['image']),
                        'steering': float(row['steering']),
                        'throttle': float(row['throttle'])
                    })
            except Exception as e:
                logger.error(f"Error reading {csv_path}: {e}")

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __iter__(self) -> Generator[Batch, None, None]:
        indices = np.arange(self.num_samples)
        if self.shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, self.num_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, self.num_samples)
            batch_indices = indices[start_idx: end_idx]

            x_batch = []
            t_batch = []

            for i in batch_indices:
                sample = self.samples[i]

                img = cv2.imread(sample['path'])
                if img is None:
                    continue

                steering = sample['steering']
                throttle = sample['throttle']
                # 左右をフリップ
                if self.shuffle and random.random() > 0.5:
                    img = cv2.flip(img, 1)
                    steering = -steering

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.astype(np.float32) / 255.0
                img = img.transpose(2, 0, 1) # (H, W, C) -> (C, H, W)

                x_batch.append(img)
                t_batch.append([steering, throttle])

            if not x_batch:
                continue

            yield Batch(
                images=np.array(x_batch),
                labels=np.array(t_batch).astype(np.float32)
            )
