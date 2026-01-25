import logging
from loader import Loader
from trainer import Trainer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> None:
    loader = Loader("./data")
    images, steerings, throttles, cfg = loader.load_sessions()

    trainer: Trainer = Trainer(images, steerings, throttles, cfg)
    trainer.train()

if __name__ == "__main__":
    main()
