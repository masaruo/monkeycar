import numpy as np
import logging
from tqdm import trange
from shared.models import ModelConfig
from shared.optimizer import Adam
from shared.network import ConvNetwork
from typing import Final
import json
import os
from datetime import datetime

logger = logging.getLogger(__name__)


EPOCHS: Final = 50
BATCH_SIZE: Final = 32
VALIDATION_SPLIT: Final = 0.2


class Trainer:
    def __init__(self, images: np.ndarray, steerings: np.ndarray, throttles: np.ndarray, cfg: ModelConfig) -> None:
        self.images: np.ndarray = images
        self.steerings: np.ndarray = steerings
        self.throttles: np.ndarray = throttles
        self.cfg: ModelConfig = cfg
        
        # Initialize DataTransformer with config
        from shared.transformer import DataTransformer
        self.transformer = DataTransformer(config=cfg)
        
        self.targets: np.ndarray = self.__normalize()
        self.network = ConvNetwork(input_dim=(3, cfg.image_size[1], cfg.image_size[0]))
        
        # Output directory setup
        self.output_dir = f"output/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info(f"Trainer Initiated. Output dir: {self.output_dir}")

    def __normalize(self) -> np.ndarray:
        # Use DataTransformer for consistent normalization
        # Now passing arrays directly as normalize_labels supports it
        s_norm, t_norm = self.transformer.normalize_labels(self.steerings, self.throttles)
        return np.column_stack([s_norm, t_norm])

    def _split_data(self, split_rate: float = VALIDATION_SPLIT) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        num_samples = len(self.images)

        indices = np.random.permutation(num_samples)
        
        val_size = int(num_samples * split_rate)
        train_size = num_samples - val_size

        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        x_train, x_test = self.images[train_indices], self.images[val_indices]
        t_train, t_test = self.targets[train_indices], self.targets[val_indices]

        return x_train, t_train, x_test, t_test

    def train(self) -> None:
        x_train, t_train, x_test, t_test = self._split_data()
        optimizer = Adam(lr=0.001)

        num_samples = x_train.shape[0]

        for epoch in trange(EPOCHS, desc="Epochs"):
            indices = np.random.permutation(num_samples)
            for i in trange(0, num_samples, BATCH_SIZE):
                batch_idx = indices[i : i + BATCH_SIZE]
                batch_x = x_train[batch_idx]
                batch_t = t_train[batch_idx]

                grads = self.network.gradient(batch_x, batch_t)

                optimizer.update(self.network.params, grads)
            
            current_loss = self.network.loss(batch_x, batch_t)
            logger.info(f"Epoch {epoch+1}/{EPOCHS} - loss: {current_loss:.4f}")

        # Save results
        self.network.save_params(os.path.join(self.output_dir, "params.pkl"))
        
        # Configの保存
        with open(os.path.join(self.output_dir, "config.json"), 'w') as f:
            f.write(self.cfg.model_dump_json(indent=4))
        
        logger.info(f"Training finished. Results saved to {self.output_dir}")
