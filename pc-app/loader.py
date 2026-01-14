import logging
from pathlib import Path
import numpy as np
import cv2
import pandas as pd
# from shared.models import ModelConfig



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
        self.normalize_param = 255.0

        self.images = []
        self.steerings = []
        self.throttles = []
        self.config = ModelConfig()

    def load_sessions(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """全セッションからデータを読み込む"""
        # session_ で始まるディレクトリを取得（名前順でソート）
        session_dirs = sorted([
            d for d in self.data_dir.iterdir()
            if d.is_dir() and d.name.startswith('session_')
        ])

        if not session_dirs:
            logger.warning(f"{self.data_dir} 内に session_ で始まるフォルダが見つかりません")
            return np.array([]), np.array([]), np.array([])

        logger.info(f"見つかったセッション数: {len(session_dirs)}")

        for session_dir in session_dirs:
            logger.info(f"読み込み中: {session_dir.name}")
            self.__load_session(session_dir)

        logger.info(f"合計読み込み件数: {len(self.images)}")

        return (
            np.array(self.images),
            np.array(self.steerings),
            np.array(self.throttles),
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

            # 前処理
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.image_size)
            img = img.astype(np.float32) / self.normalize_param


            # データ追加
            self.images.append(img)
            self.steerings.append(float(row['steering']))
            self.throttles.append(float(row['throttle']))
            loaded_count += 1

        logger.info(f"  {session_dir.name}: {loaded_count}/{len(df)} 件読み込み成功")

    def get_stats(self) -> dict:
        """データセット統計を取得"""
        if not self.images:
            return {
                'num_samples': 0,
                'error': 'データが読み込まれていません'
            }

        steerings = np.array(self.steerings)
        throttles = np.array(self.throttles)

        config = {
            "image_size": self.image_size,
            'image_shape': (len(self.images),) + self.image_size + (3,),
            
            'steering_min': float(steerings.min()),
            'steering_max': float(steerings.max()),
            'steering_mean': float(steerings.mean()),
            'steering_std': float(steerings.std()),
            'throttle_min': float(throttles.min()),
            'throttle_max': float(throttles.max()),
            'throttle_mean': float(throttles.mean()),
            'throttle_std': float(throttles.std()),
            'num_samples': len(self.images),
        }
        return {
        }


if __name__ == '__main__':
    # ロガー初期化（これがないとログが出力されない）
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )

    # データ読み込みテスト
    loader = Loader('../../data')
    images, steerings, throttles = loader.load_sessions()

    # 統計表示
    print("\n=== データセット統計 ===")
    stats = loader.get_stats()
    for key, val in stats.items():
        print(f"{key:20s}: {val}")

    # データ品質チェック
    num_samples = stats.get('num_samples', 0)
    if num_samples == 0:
        print("\nエラー: データが読み込まれていません")
    elif num_samples < 1000:
        print(f"\n警告: サンプル数が少ないです（現在{num_samples}件、最低1000件推奨）")

    steering_mean = stats.get('steering_mean', 0)
    if abs(steering_mean) > 0.3:
        print(f"警告: ステアリングが偏っています（平均={steering_mean:.2f}）")

    steering_std = stats.get('steering_std', 0)
    if steering_std < 0.05:
        print("警告: ステアリング変動が小さい（直進ばかり？）")
