import logging
from loader import Loader

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    print("Hello from pc-app!")


if __name__ == "__main__":
    main()
