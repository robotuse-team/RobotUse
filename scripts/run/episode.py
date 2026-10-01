#!/usr/bin/env python3
"""Run one RobotUse episode with the configured agents and native simulator."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.runtime.cli import main, resolve
from src.utils.logging_utils import configure_logging


if __name__ == '__main__':
    configure_logging()
    raise SystemExit(main())
