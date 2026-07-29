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
    """(Re)configure the logger.

    Note: `logger = setup_logger()` at the bottom of this module runs once as soon as
    `utils.logger` is imported (with default args, writing to DEFAULT_LOG_FILE). Later,
    ZxAgent.__init__ in main.py calls setup_logger() again with the log_file/level from
    the config file. This used to just `return logger` early if handlers already existed,
    which meant the second call never actually took effect -- no matter what logging.file/
    level the config file specified, the logger kept using the defaults from the very
    first import-time call (logs/agent.log, INFO). Now every call clears the old
    handlers and re-binds with the newly passed-in parameters, so logging.file/level
    from the config file actually takes effect.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()

    log_path = Path(log_file or DEFAULT_LOG_FILE)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(log_path, maxBytes=max_bytes, backupCount=backup_count)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s"))
    logger.addHandler(file_handler)
    return logger


logger = setup_logger()
