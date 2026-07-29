"""
Logging utility

Logs only go to a file, never to the terminal -- the terminal is reserved for the
recognized text / AI answers (printed via print() in main.py), so logs don't get
mixed in with the actual transcription and make the screen hard to read.
For debugging, open the file pointed to by log_file (default logs/agent.log), or
tail -f it in real time.
"""

import logging
from pathlib import Path
from logging.handlers import RotatingFileHandler

DEFAULT_LOG_FILE = "logs/agent.log"


def setup_logger(name="agent", level="INFO", log_file=None, max_bytes=10485760, backup_count=3):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    if logger.handlers:
        return logger
    log_path = Path(log_file or DEFAULT_LOG_FILE)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(log_path, maxBytes=max_bytes, backupCount=backup_count)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s"))
    logger.addHandler(file_handler)
    return logger


logger = setup_logger()
