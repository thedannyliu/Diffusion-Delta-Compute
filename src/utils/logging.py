import logging
import os
from typing import Optional


def init_run_logging(out_dir: str, run_name: str, level: int = logging.INFO) -> None:
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, f"{run_name}.log")
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ],
    )


def get_logger(name: Optional[str] = None) -> logging.Logger:
    return logging.getLogger(name)


