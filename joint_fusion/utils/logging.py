import logging
import os
import sys


def setup_logging(
    log_dir: str,
    log_level: str = "INFO",
    log_name: str = "train.log",
):
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, log_name)

    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))

    logger.handlers.clear()  # avoid duplicate logs

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    # ---- file handler ----
    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)

    # ---- console handler ----
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger
