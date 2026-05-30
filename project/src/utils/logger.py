import logging
import os

from src.utils.config import settings


os.makedirs(
    settings.LOG_DIR,
    exist_ok=True,
)

LOG_FILE = os.path.join(
    settings.LOG_DIR,
    "rag_system.log",
)


logger = logging.getLogger("rag_system")

logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(message)s"
)


# === Console Handler ===

console_handler = logging.StreamHandler()

console_handler.setLevel(logging.INFO)

console_handler.setFormatter(formatter)


# === File Handler ===

file_handler = logging.FileHandler(
    LOG_FILE,
    encoding="utf-8",
)

file_handler.setLevel(logging.INFO)

file_handler.setFormatter(formatter)


logger.addHandler(console_handler)
logger.addHandler(file_handler)