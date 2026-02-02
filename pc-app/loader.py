import logging
import math
from typing import NamedTuple, List, Optional, Generator
import random
from pathlib import Path
import numpy as np
import cv2
import pandas as pd
from cnn.models import ModelConfig
from cnn.transformer import DataTransformer

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

        self.samples = []

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
            if not csv_path.exists:
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

# class Loader:
#     """訓練データの読み込みと前処理"""

#     def __init__(
#         self,
#         data_dir: str = "./data",
#         image_size: tuple[int, int] = (160, 120),
#     ):
#         """
#         Args:
#             data_dir: データセットディレクトリ (複数のsessionフォルダを含む)
#             image_size: リサイズ対象のサイズ (幅, 高さ)
#         """
#         self.data_dir = Path(data_dir).resolve()
#         self.image_size = image_size
#         self.norm_divisor = 255.0

#         self.images = []
#         self.steerings = []
#         self.throttles = []

#         self.transformer = DataTransformer(image_size=image_size)

#     def load_sessions(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, ModelConfig | None]:
#         """全セッションからデータを読み込む"""
#         # session_ で始まるディレクトリを取得（名前順でソート）
#         session_dirs = sorted([
#             d for d in self.data_dir.iterdir()
#             if d.is_dir() and d.name.startswith('session_')
#         ])

#         if not session_dirs:
#             logger.warning(f"{self.data_dir} 内に session_ で始まるフォルダが見つかりません")
#             return np.array([]), np.array([]), np.array([]), None

#         logger.info(f"見つかったセッション数: {len(session_dirs)}")

#         for session_dir in session_dirs:
#             logger.info(f"読み込み中: {session_dir.name}")
#             self.__load_session(session_dir)

#         logger.info(f"合計読み込み件数: {len(self.images)}")


#         steerings_nparr = np.array(self.steerings)
#         throttles_nparr = np.array(self.throttles)
#         return (
#             np.array(self.images),
#             steerings_nparr,
#             throttles_nparr,
#             self.__get_config(steerings_nparr, throttles_nparr)
#         )

#     def __load_session(self, session_dir: Path) -> None:
#         """単一セッションフォルダからデータを読み込む"""
#         csv_path = session_dir / 'records.csv'
#         image_dir = session_dir / 'image'

#         # CSVファイルの存在確認
#         if not csv_path.exists():
#             logger.warning(f"records.csvが見つかりません: {csv_path}")
#             return

#         # 画像ディレクトリの存在確認
#         if not image_dir.exists():
#             logger.warning(f"imageディレクトリが見つかりません: {image_dir}")
#             return

#         df = pd.read_csv(csv_path)
#         loaded_count = 0

#         for _, row in df.iterrows():
#             image_filename = row['image']
#             image_path = image_dir / image_filename

#             # 画像ファイルの存在確認
#             if not image_path.exists():
#                 logger.warning(f"画像ファイルが見つかりません: {image_path}")
#                 continue

#             # 画像読み込み
#             img = cv2.imread(str(image_path))
#             if img is None:
#                 logger.warning(f"画像の読み込みに失敗: {image_path}")
#                 continue

#             # 前処理 (DataTransformerを使用)
#             # RGB変換のみここで行い、あとはTransformerに任せる
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             img = self.transformer.transform_image(img)

#             # データ追加
#             self.images.append(img)
#             self.steerings.append(float(row['steering']))
#             self.throttles.append(float(row['throttle']))
#             loaded_count += 1

#         logger.info(f"  {session_dir.name}: {loaded_count}/{len(df)} 件読み込み成功")

#     def __get_config(self, steerings: np.ndarray, throttles: np.ndarray) -> ModelConfig:
#         data = {
#             'image_size': self.image_size,
#             'norm_divisor': self.norm_divisor,
#             'steering_min': float(steerings.min()),
#             'steering_max': float(steerings.max()),
#             'throttle_min': float(throttles.min()),
#             'throttle_max': float(throttles.max()),
#         }
#         return ModelConfig(**data)
