import logging

from typing import Optional
import sys

import logging

from typing import Optional
import sys


def setup_logging(log_file: Optional[str] = None):
    """Setup logging with colorful and informational formatting"""

    # Define color codes
    class LogColors:
        RESET = "\033[0m"
        BOLD = "\033[1m"
        DIM = "\033[2m"

        # Levels
        DEBUG = "\033[36m"  # Cyan
        INFO = "\033[32m"  # Green
        WARNING = "\033[33m"  # Yellow
        ERROR = "\033[31m"  # Red
        CRITICAL = "\033[35m"  # Magenta

        # Components
        TIME = "\033[90m"  # Gray
        NAME = "\033[94m"  # Blue
        SEPARATOR = "\033[90m"  # Gray

    class ColoredFormatter(logging.Formatter):
        """Custom formatter with colors and aesthetic layout"""

        LEVEL_COLORS = {
            logging.DEBUG: LogColors.DEBUG,
            logging.INFO: LogColors.INFO,
            logging.WARNING: LogColors.WARNING,
            logging.ERROR: LogColors.ERROR,
            logging.CRITICAL: LogColors.CRITICAL,
        }

        def format(self, record):
            # Add color to level name (center-aligned, padded to 8 chars)
            level_color = self.LEVEL_COLORS.get(record.levelno, "")
            record.levelname = f"{level_color}{record.levelname:^8s}{LogColors.RESET}"

            # Color the timestamp
            record.asctime = f"{LogColors.TIME}{self.formatTime(record, '%H:%M:%S')}{LogColors.RESET}"

            # Color the logger name
            record.name = f"{LogColors.NAME}{record.name}{LogColors.RESET}"

            # Format with separator
            separator = f"{LogColors.SEPARATOR}│{LogColors.RESET}"

            return f"{record.asctime} {separator} {record.levelname}{separator} {record.getMessage()}"

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Remove existing handlers
    logger.handlers.clear()

    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(ColoredFormatter())
    logger.addHandler(console_handler)

    # File handler without colors (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "%(asctime)s │ %(levelname)-8s │ %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger
