"""Repository paths independent of the calling working directory."""
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
