"""logger."""

from __future__ import annotations

import datetime
import logging
import time

LOGGER_NAME = 'ptfid'


def get_logger(
    format: str = '%(asctime)s | %(name)s | %(levelname)s | %(funcName)s || %(message)s',
    logging_level: int = logging.WARNING,
    filename: str | None = None,
    auxiliary_handlers: list | None = None,
) -> logging.Logger:
    """Create/return logger.

    Args:
    ----
        format (str, optional): logging format.
            Default: '%(asctime)s | %(name)s | %(levelname)s | %(funcName)s | - %(message)s'.
        logging_level (int): Logging level. Default: logging.WARNING.
        filename (str, optional): If specified, save logs to a file. Deafult: None.
        auxiliary_handlers (list, optional): Other user-defined handlers. Default: None

    Returns:
    -------
        logging.Logger: logger object.

    """
    logger = logging.getLogger(LOGGER_NAME)

    if len(logger.handlers) > 0:
        return logger

    # Logging level is configured at each handler, so we set to DEBUG here.
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(format)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging_level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if filename is not None:
        file_handler = logging.FileHandler(filename, mode='a')
        file_handler.setLevel(logging.DEBUG)  # Always log everything to a file.
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if auxiliary_handlers:
        for handler in auxiliary_handlers:
            logger.addHandler(handler)

    return logger


def set_logging_level(level: int) -> None:
    """Set logging level for StreamHandler.

    Args:
    ----
        name (str): name of the logger.
        level (int): logging level.

    """
    logger = get_logger()
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setLevel(level)


class Timer:
    """Simple timer."""

    def __init__(self) -> None:
        """Construct timer."""
        self._start = None

    def start(self):
        """Start timer."""
        self._start = time.perf_counter()

    def done(self, timedelta: bool = True) -> float | datetime.timedelta:
        """End timer and return duration.

        Args:
        ----
            timedelta (bool): return duration as timedelta object.

        Returns:
        -------
            float | datetime.timedelta: duration.

        """
        assert self._start is not None, 'Must call .start() before calling this method.'

        duration = time.perf_counter() - self._start

        if timedelta:
            return datetime.timedelta(seconds=duration)

        self._start = None
        return duration
