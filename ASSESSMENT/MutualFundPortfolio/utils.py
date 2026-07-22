"""Reusable utility helpers for the Mutual Fund Portfolio project.

This module centralises cross-cutting concerns such as logging configuration,
directory management and small numeric helpers so that the rest of the code
base stays free of duplicated boilerplate.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Optional

# --------------------------------------------------------------------------- #
# Project paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT: str = os.path.dirname(os.path.abspath(__file__))
DATA_DIR: str = os.path.join(PROJECT_ROOT, "data")
REPORTS_DIR: str = os.path.join(PROJECT_ROOT, "reports")
CHARTS_DIR: str = os.path.join(PROJECT_ROOT, "charts")
LOGS_DIR: str = os.path.join(PROJECT_ROOT, "logs")

# Number of trading days used to annualise returns and NAV statistics.
TRADING_DAYS_PER_YEAR: int = 252
# Risk-free rate assumption used for the simplified Sharpe ratio.
RISK_FREE_RATE: float = 0.06


def ensure_directories(*paths: str) -> None:
    """Create every directory in ``paths`` if it does not already exist.

    Args:
        *paths: One or more absolute directory paths to create.
    """
    for path in paths:
        os.makedirs(path, exist_ok=True)


def setup_logger(
    name: str = "fund_portfolio",
    log_dir: str = LOGS_DIR,
    level: int = logging.INFO,
) -> logging.Logger:
    """Configure and return a logger that writes to both a file and console.

    The log file is timestamped so every execution produces its own log,
    satisfying the requirement to generate execution logs.

    Args:
        name: Logger name.
        log_dir: Directory where the log file is written.
        level: Logging level.

    Returns:
        A configured :class:`logging.Logger` instance.
    """
    ensure_directories(log_dir)
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid attaching duplicate handlers when the logger is requested twice.
    if logger.handlers:
        return logger

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"execution_{timestamp}.log")

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = False

    logger.info("Logger initialised. Writing logs to: %s", log_file)
    return logger


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide two numbers, returning ``default`` when the denominator is zero.

    Args:
        numerator: The dividend.
        denominator: The divisor.
        default: Value returned when ``denominator`` is zero.

    Returns:
        The quotient or ``default`` when division is undefined.
    """
    if denominator == 0 or denominator is None:
        return default
    return numerator / denominator


def format_currency(value: float, symbol: str = "\u20b9") -> str:
    """Format a numeric value as an Indian-rupee currency string.

    Args:
        value: The numeric amount.
        symbol: Currency symbol to prefix.

    Returns:
        A human readable currency string, e.g. ``"₹1,23,456.78"``.
    """
    try:
        return f"{symbol}{value:,.2f}"
    except (TypeError, ValueError):
        return f"{symbol}0.00"


def get_timestamp() -> str:
    """Return the current timestamp formatted for filenames and reports.

    Returns:
        A timestamp string, e.g. ``"2026-07-21 10:15:30"``.
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def percentage(part: float, whole: float, default: float = 0.0) -> float:
    """Return ``part`` as a percentage of ``whole``.

    Args:
        part: The portion value.
        whole: The total value.
        default: Value returned when ``whole`` is zero.

    Returns:
        The percentage represented by ``part`` relative to ``whole``.
    """
    return safe_divide(part, whole, default) * 100.0
