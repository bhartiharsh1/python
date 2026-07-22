"""Entry point for the Mutual Fund Portfolio Performance & Risk Analysis tool.

Running this module executes the full pipeline in order:

1. Load data
2. Clean data
3. Merge data
4. Perform analysis (statistics, investors, funds, financial metrics)
5. Generate charts
6. Export reports
7. Print a console summary

Execution logs are written to the ``logs`` directory.

Usage::

    python main.py
"""

from __future__ import annotations

import os
import sys

from data_loader import DataLoadError
from fund_portfolio import FundPortfolio
from utils import DATA_DIR, setup_logger


def _configure_utf8_output() -> None:
    """Force stdout/stderr to UTF-8 so currency symbols never crash the app.

    On Windows the default console encoding (cp1252) cannot encode the rupee
    symbol, which raises ``UnicodeEncodeError`` when output is redirected or
    piped. Reconfiguring the streams makes console output robust everywhere.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def _ensure_sample_data(logger) -> None:
    """Generate sample data automatically if the CSV files are missing.

    Args:
        logger: Configured logger instance.
    """
    required = ["investors.csv", "funds.csv", "transactions.csv", "nav_history.csv"]
    missing = [
        name for name in required if not os.path.exists(os.path.join(DATA_DIR, name))
    ]
    if missing:
        logger.warning(
            "Missing data files %s. Generating sample data...", ", ".join(missing)
        )
        import generate_data

        generate_data.main()


def main() -> int:
    """Run the full portfolio-analysis pipeline.

    Returns:
        Process exit code: ``0`` on success, ``1`` on failure.
    """
    _configure_utf8_output()
    logger = setup_logger()
    logger.info("Application started")
    try:
        _ensure_sample_data(logger)
        portfolio = FundPortfolio(logger=logger)
        portfolio.run_full_pipeline()
        logger.info("Application finished successfully")
        return 0
    except DataLoadError as exc:
        logger.error("Data loading failed: %s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - top-level guard for CLI robustness.
        logger.exception("Unexpected error during execution: %s", exc)
        print(f"UNEXPECTED ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
