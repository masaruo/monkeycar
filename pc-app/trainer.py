import os
import numpy as np
from loader import BatchLoader
from cnn.network import DeepConvNet
from cnn.optimizer import Adam
from tqdm import trange
import pickle
import logging
from typing import Final
from cnn.models import ModelConfig
from pathlib import Path


TRAIN_SESSIONS: Final = [
    'session_shortcut_train',
]

TEST_SESSIONS: Final = [
    'session_shortcut_test',
]


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class Trainer:
    def __init__(
        self,
        data_dir="../data",
        output_dir="weights_bin",
        image_size=(160, 120),
        batch_size=32,
        learning_rate=0.01
        ) -> None:
        """
        学習管理クラス

        Args:
            data_dir (str): 学習データ(sessionフォルダ)があるルートディレクトリ
            output_dir (str): 重みの保存先
            image_size (tuple): 画像サイズ (Width, Height)
            batch_size (int): ミニバッチサイズ
            learning_rate (float): 学習率
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.optimizer = Adam(learning_rate)

        # # 内部状態
        # self.network = None
        # self.x_train = None
        # self.t_train = None
        # self.x_test = None
        # self.t_test = None

        # データのロード
        self._prepare_data()

        # ネットワークの初期化
        self._init_network()


    def _prepare_data(self):
        train_sessions = [Path(s).name for s in TRAIN_SESSIONS]
        self.train_loader = BatchLoader(
            data_dir=self.data_dir,
            target_sessions=train_sessions,
            batch_size=self.batch_size,
            shuffle=True
        )

        test_sessions = [Path(s).name for s in TEST_SESSIONS]
        self.test_loader = BatchLoader(
            data_dir=self.data_dir,
            target_sessions=test_sessions,
            batch_size=self.batch_size,
            shuffle=False
        )
        # """データの読み込みと前処理"""
        # print(f"Loading data from {self.data_dir}...")
        # loader = Loader(data_dir=self.data_dir, image_size=self.image_size)

        # # config も受け取る
        # images, steerings, throttles, config = loader.load_sessions()

        # if len(images) == 0:
        #     raise RuntimeError("No data found. Check your data path.")

        # # Transformer の初期化
        # self.transformer = DataTransformer(config=config, image_size=self.image_size)

        # # 1. 画像の整形 & 正規化
        # # (N, H, W, C) -> (N, C, H, W)
        # if images.ndim == 4 and images.shape[3] == 3:
        #     images = images.transpose(0, 3, 1, 2)

        # # 正規化 (0.0 ~ 1.0)
        # self.x_all = images.astype('float32') / 255.0

        # # 2. 正解ラベルの正規化
        # t_steer_norm, t_throt_norm = self.transformer.normalize_labels(steerings, throttles)

        # self.t_all = np.column_stack((t_steer_norm, t_throt_norm)).astype('float32')

        # # ============================================================
        # # 【追加箇所】 データの左右反転による水増し
        # # ============================================================
        # print(f"Original samples: {len(self.x_all)}")
        # print("Augmenting data with horizontal flips...")

        # # 画像を左右反転: (N, C, H, W) なので、最後のW軸(インデックス3)を反転
        # x_flipped = self.x_all[:, :, :, ::-1]

        # # 正解ラベルの反転: ステアリングの符号を反転 (Throttleはそのまま)
        # t_flipped = self.t_all.copy()
        # t_flipped[:, 0] = -t_flipped[:, 0]  # Steering is index 0

        # # オリジナルと結合
        # self.x_all = np.concatenate([self.x_all, x_flipped], axis=0)
        # self.t_all = np.concatenate([self.t_all, t_flipped], axis=0)

        # print(f"Data augmentation finished. Total samples: {len(self.x_all)}")

        # # 4. Train/Test分割 (8:2)
        # data_size = self.x_all.shape[0]
        # indices = np.random.permutation(data_size)
        # split_idx = int(data_size * 0.8)

        # train_idx, test_idx = indices[:split_idx], indices[split_idx:]

        # self.x_train = self.x_all[train_idx]
        # self.x_test = self.x_all[test_idx]
        # self.t_train = self.t_all[train_idx]
        # self.t_test = self.t_all[test_idx]

        # print(f"Data loaded. Train: {len(self.x_train)}, Test: {len(self.x_test)}")
        # print(f"Input shape: {self.x_train.shape[1:]}")
        # print(f"Target Min: {self.t_train.min(axis=0)}")
        # print(f"Target Max: {self.t_train.max(axis=0)}")
        # print(f"Target Mean: {self.t_train.mean(axis=0)}")


    def _init_network(self):
            """CNNの構築"""
            input_dim = (3, self.image_size[0], self.image_size[1])

            # DeepConvNet の初期化
            self.network = DeepConvNet(
                input_dim=input_dim,

                # 1層目: フィルター16枚, サイズ3x3
                conv_param_1={'filter_num': 16, 'filter_size': 3, 'pad': 1, 'stride': 2},

                # 2層目: フィルター32枚, サイズ3x3
                conv_param_2={'filter_num': 32, 'filter_size': 3, 'pad': 1, 'stride': 1},

                hidden_size=100,
                output_size=2
            )

            self.optimizer = Adam(lr=0.0001)

    def train(self, epochs: int=5000):
        logging.info(f"\nStart Training for {epochs} iterations...")
        for epoch in trange(epochs):
            loss_sum = 0
            count = 0
            
            for batch in self.train_loader:
                grads = self.network.gradient(batch.images, batch.labels)
                
                self.optimizer.update(self.network.params, grads)
                
                loss = self.network.loss(batch.images, batch.labels)
                loss_sum += loss
                count += 1
            
            avg_loss = loss_sum / count
            logger.info(f"Epoch {epoch+1:2d}/{epochs} | Train Loss: {avg_loss:.5f}")
            
            self.evaluate()

        self.save_weights()

    def evaluate(self):
        total_loss = 0
        total_samples = 0
        
        for batch in self.test_loader:
            loss = self.network.loss(batch.images, batch.labels)
            
            batch_len = len(batch.images)
            total_loss = loss * batch_len
            total_samples += batch_len
        
        final_loss = total_loss / total_samples if total_samples > 0 else 0
        logger.info(f"   >>> Test Loss: {final_loss:.5f}")

    def save_weights(self, param_filename="params.pkl", config_filename="config.json"):
        os.makedirs(self.output_dir, exist_ok=True)

        # 1. パラメータの保存 (params.pkl)
        param_path = os.path.join(self.output_dir, param_filename)
        with open(param_path, 'wb') as f:
            pickle.dump(self.network.params, f)

        all_steerings = []
        all_throttles = []
        if hasattr(self.train_loader, 'samples') and self.train_loader.samples:
            all_steerings = [s['steering'] for s in self.train_loader.samples]
            all_throttles = [s['throttle'] for s in self.train_loader.samples]

        config = ModelConfig(
            image_size = self.image_size,
            steering_min = float(min(all_steerings)) if all_steerings else 0.0,
            steering_max = float(max(all_steerings)) if all_steerings else 0.0,
            throttle_min = float(min(all_throttles)) if all_throttles else 0.0,
            throttle_max = float(max(all_throttles)) if all_throttles else 0.0,
            num_samples=len(self.train_loader.samples) if hasattr(self.train_loader, 'samples') else 0
        )

        config_path = os.path.join(self.output_dir, config_filename)
        
        with open(config_path, 'w') as f:
            # Pydantic v2 の場合 (推奨)
            f.write(config.model_dump_json(indent=2))
            
            # もし Pydantic v1 (古いバージョン) でエラーが出る場合はこちら:
            # f.write(config.json(indent=2))
            
        print(f"Saved config to {config_path}")
        # 2. 設定ファイルの保存 (config.json)
        # DataTransformerが持っているconfigを取り出して保存します
        # if self.transformer.config is not None:
        #     config_path = os.path.join(self.output_dir, config_filename)
        
        #     with open(config_path, 'w') as f:
        #         # Pydantic v2対応 (v1の場合は .json())
        #         f.write(self.transformer.config.model_dump_json(indent=2))
        #     print(f"Saved config to {config_path}")
        # else:
        #     print("Warning: Config object is missing. config.json was not saved.")

if __name__ == "__main__":
    # 使用例
    try:
        # ディレクトリは環境に合わせて修正してください
        trainer = Trainer(data_dir="./data", batch_size=32, learning_rate=0.0001)

        # 学習実行
        trainer.train(epochs=20)

        # 重み保存
        trainer.save_weights()

    except Exception as e:
        print(f"An error occurred: {e}")
