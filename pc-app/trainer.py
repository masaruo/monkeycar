import os
from loader import BatchLoader
from cnn.network import CarConvNet
from cnn.optimizer import Adam, SGD, Optimizer
from cnn.models import ModelConfig
from cnn.transformer import DataTransformer
from tqdm import tqdm, trange
import logging
from typing import Final
from pathlib import Path


TRAIN_SESSIONS: Final = [
    'session_1770198487',
    'session_1770198509',
]

TEST_SESSIONS: Final = [
    'session_1770198529',
]

log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, log_level), format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        optimizer: type[Optimizer],
        data_dir: str="data",
        output_dir: str="models",
        image_size: tuple[int, int]=(160, 120),
        batch_size: int=32,
        learning_rate: float=0.0001,
        ) -> None:
        """
        学習管理クラス

        Args:
            data_dir (str): 学習データ(sessionフォルダ)があるルートディレクトリ
            output_dir (str): 重みの保存先
            image_size (tuple): 画像サイズ (Width, Height)
            batch_size (int): ミニバッチサイズ
            learning_rate (float): 学習率
            optimizer (type): Optimizer class (e.g., Adam, SGD)
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.optimizer = optimizer(learning_rate)
        self.final_loss = 0.0
        self.final_val_loss = 0.0

        # データのロード
        self._prepare_data()

        # ネットワークの初期化
        self._init_network()


    def _prepare_data(self) -> None:
        config = ModelConfig(
            image_size=self.image_size,
            steering_min = -1.0,
            steering_max = 1.0,
            throttle_min = 0.0,
            throttle_max = 1.0,
        )

        self.tranformer = DataTransformer(config=config)

        train_sessions = [Path(s).name for s in TRAIN_SESSIONS]
        self.train_loader = BatchLoader(
            data_dir=self.data_dir,
            target_sessions=train_sessions,
            batch_size=self.batch_size,
            shuffle=True,
            transformer=self.tranformer,
        )

        test_sessions = [Path(s).name for s in TEST_SESSIONS]
        self.test_loader = BatchLoader(
            data_dir=self.data_dir,
            target_sessions=test_sessions,
            batch_size=self.batch_size,
            shuffle=False,
            transformer=self.tranformer,
        )

    def _init_network(self) -> None:
            """CNNの構築"""
            input_dim = (3, self.image_size[1], self.image_size[0]) #(3, 120, 160)

            self.network = CarConvNet(
                input_dim=input_dim,
                hidden_size=100,
                output_size=2,
            )

    def train(self, epochs: int=20) -> None:
        logging.info(f"\nStart Training for {epochs} iterations...")
        epoch_bar = trange(epochs, desc="Training")

        for epoch in epoch_bar:
            self.network.train_mode()

            loss_sum = 0.0
            count = 0

            batch_bar = tqdm(self.test_loader, leave=False, desc=f"{epoch+1}")
            for batch in batch_bar:
                self.network.gradient(batch.images, batch.labels)
                self.optimizer.update(self.network.params())

                loss = self.network.loss(batch.images, batch.labels)
                loss_sum += loss
                count += 1
                batch_bar.set_postfix({"loss": f"{loss:.4f}"})

            avg_loss = loss_sum / count if count > 0 else 0
            self.final_loss = avg_loss

            self.evaluate()

            tqdm.write(f"Epoch {epoch+1:2d} | Train: {self.final_loss:.4f} | Test: {self.final_val_loss:.4f}")
            epoch_bar.set_postfix({
                'train_loss': f"{self.final_loss:.4f}",
                'val_loss': f"{self.final_val_loss:.4f}",
            })

        self.save_weights()

    def evaluate(self) -> None:
        self.network.eval_mode()

        total_loss = 0.0
        total_samples = 0

        for batch in self.test_loader:
            loss = self.network.loss(batch.images, batch.labels)

            batch_len = len(batch.images)
            total_loss += loss * batch_len
            total_samples += batch_len

        avg_val_loss = total_loss / total_samples if total_samples > 0 else 0
        self.final_val_loss = avg_val_loss

    def save_weights(self, param_filename: str="params.pkl", config_filename: str="config.json") -> None:
        os.makedirs(self.output_dir, exist_ok=True)

        param_path = os.path.join(self.output_dir, param_filename)
        self.network.save_params(param_path)

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
            num_samples = len(self.train_loader.samples) if hasattr(self.train_loader, 'samples') else 0,
            final_loss = self.final_loss,
            final_val_loss = self.final_val_loss,
        )

        config_path = os.path.join(self.output_dir, config_filename)

        with open(config_path, 'w') as f:
            f.write(config.model_dump_json(indent=2))

        logger.info(f"Saved weights to {param_path}, config to {config_path}")

if __name__ == "__main__":
    try:
        trainer = Trainer(data_dir="./data", batch_size=32, learning_rate=0.0001, optimizer=Adam)
        # trainer = Trainer(data_dir="./data", batch_size=32, learning_rate=0.01, optimizer=SGD)
        trainer.train(epochs=50)
        trainer.save_weights()

    except Exception as e:
        print(f"An error occurred: {e}")
