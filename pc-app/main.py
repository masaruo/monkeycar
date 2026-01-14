import logging
from loader import Loader
from trainer import Trainer
import numpy as np
from shared.models import ModelConfig

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    loader = Loader("./data")
    images, steerings, throttles, cfg = loader.load_sessions()

    trainer: Trainer = Trainer(images, steerings, throttles, cfg)
    trainer.train()

if __name__ == "__main__":
    main()
