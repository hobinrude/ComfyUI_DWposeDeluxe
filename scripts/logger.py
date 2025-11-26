# ComfyUI_DWposeDeluxe/scripts/logger.py

import os
import sys
import copy
import logging
from enum import Enum

class TEXTS(Enum):
    LOGGER_PREFIX = "DWposeNode"

class ColoredFormatter(logging.Formatter):
    COLORS = {
        "INFO": "\033[0;32m",      # GREEN
        "WARNING": "\033[93m",     # YELLOW
        "ERROR": "\033[91m",       # RED
        "RESET": "\033[0m",        # RESET COLOR
    }

    def format(self, record):
        colored_record = copy.copy(record)
        levelname = colored_record.levelname
        seq = self.COLORS.get(levelname, self.COLORS["RESET"])
        colored_record.levelname = f"{seq}{levelname}{self.COLORS['RESET']}"
        return super().format(colored_record)

logger = logging.getLogger(TEXTS.LOGGER_PREFIX.value)
logger.propagate = False

if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ColoredFormatter("[%(name)s][%(levelname)s] %(message)s"))
    logger.addHandler(handler)

logger.setLevel(logging.INFO)
