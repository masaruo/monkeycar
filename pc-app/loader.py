import logging
from pathlib import Path
import numpy as np
import cv2
import pandas as pd
from shared.models import ModelConfig

logger = logging.getLogger(__name__)


class Loader:
    """訓練データの読み込みと前処理"""

    def __init__(
        self,
        data_dir: str = "data",
        image_size: tuple[int, int] = (160, 120),
    ):
        """
        Args:
            data_dir: データセットディレクトリ (複数のsessionフォルダを含む)
            image_size: リサイズ対象のサイズ (幅, 高さ)
        """
        self.data_dir = Path(data_dir).resolve()
        self.image_size = image_size
        self.norm_divisor = 255.0

        self.images = []
        self.steerings = []
        self.throttles = []
        
        # DataTransformer (画像処理モード用)
        from shared.transformer import DataTransformer
        self.transformer = DataTransformer(image_size=image_size)

    def load_sessions(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, ModelConfig | None]:
        """全セッションからデータを読み込む"""
        # session_ で始まるディレクトリを取得（名前順でソート）
        session_dirs = sorted([
            d for d in self.data_dir.iterdir()
            if d.is_dir() and d.name.startswith('session_')
        ])

        if not session_dirs:
            logger.warning(f"{self.data_dir} 内に session_ で始まるフォルダが見つかりません")
            return np.array([]), np.array([]), np.array([]), None

        logger.info(f"見つかったセッション数: {len(session_dirs)}")

        for session_dir in session_dirs:
            logger.info(f"読み込み中: {session_dir.name}")
            self.__load_session(session_dir)

        logger.info(f"合計読み込み件数: {len(self.images)}")


        steerings_nparr = np.array(self.steerings)
        throttles_nparr = np.array(self.throttles)
        return (
            np.array(self.images),
            steerings_nparr,
            throttles_nparr,
            self.__get_config(steerings_nparr, throttles_nparr)
        )

    def __load_session(self, session_dir: Path) -> None:
        """単一セッションフォルダからデータを読み込む"""
        csv_path = session_dir / 'records.csv'
        image_dir = session_dir / 'image'

        # CSVファイルの存在確認
        if not csv_path.exists():
            logger.warning(f"records.csvが見つかりません: {csv_path}")
            return

        # 画像ディレクトリの存在確認
        if not image_dir.exists():
            logger.warning(f"imageディレクトリが見つかりません: {image_dir}")
            return

        df = pd.read_csv(csv_path)
        loaded_count = 0

        for _, row in df.iterrows():
            image_filename = row['image']
            image_path = image_dir / image_filename

            # 画像ファイルの存在確認
            if not image_path.exists():
                logger.warning(f"画像ファイルが見つかりません: {image_path}")
                continue

            # 画像読み込み
            img = cv2.imread(str(image_path))
            if img is None:
                logger.warning(f"画像の読み込みに失敗: {image_path}")
                continue

            # 前処理 (DataTransformerを使用)
            # RGB変換のみここで行い、あとはTransformerに任せる
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = self.transformer.transform_image(img)

            # データ追加
            self.images.append(img)
            self.steerings.append(float(row['steering']))
            self.throttles.append(float(row['throttle']))
            loaded_count += 1

        logger.info(f"  {session_dir.name}: {loaded_count}/{len(df)} 件読み込み成功")

    def __get_config(self, steerings: np.ndarray, throttles: np.ndarray) -> ModelConfig:
        data = {
            'image_size': self.image_size,
            'norm_divisor': self.norm_divisor,
            'steering_min': float(steerings.min()),
            'steering_max': float(steerings.max()),
            'throttle_min': float(throttles.min()),
            'throttle_max': float(throttles.max()),
        }
        return ModelConfig(**data)
