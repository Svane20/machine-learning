import atexit
import functools
import logging
import sys
from typing import TextIO

from .train_utils import makedir


def setup_logging(name: str, out_directory: str = None) -> None:
    """
    Setup logging for the training process.

    Args:
        name (str): Name of the logger.
        out_directory (str): Output directory for the logs.

    Returns:
        logging.Logger: Logger object.
    """
    log_filename = None
    if out_directory:
        makedir(out_directory)
        log_filename = f"{out_directory}/log.txt"

    logger = logging.getLogger(name)
    logger.setLevel("INFO")

    # Create formatter
    FORMAT = "%(levelname)s %(asctime)s %(filename)s:%(lineno)4d: %(message)s"
    formatter = logging.Formatter(FORMAT)

    # Cleanup any existing handlers
    for h in logger.handlers:
        logger.removeHandler(h)
    logger.root.handlers = []

    # Set up the console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel("INFO")
    logger.addHandler(console_handler)

    if log_filename:
        file_handler = logging.StreamHandler(_cached_log_stream(log_filename))
        file_handler.setLevel("INFO")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logging.root = logger


@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename: str) -> TextIO:
    """
    Create a log stream with a buffer size of 10 KB.

    Args:
        filename (str): Name of the log file.

    Returns:
        TextIO: Log stream.
    """
    log_buffer_kb = 10 * 1024  # 10 KB
    io = open(filename, "a", buffering=log_buffer_kb)
    atexit.register(io.close)
    return io
