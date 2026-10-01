"""Disposable UI cache of frames already captured by the episode recorder."""
from __future__ import annotations

import base64
from io import BytesIO
import json
from pathlib import Path
import time

from PIL import Image

from src.utils.logging_utils import get_logger


class LivePreview:
    """Publish a paired image snapshot without rendering or advancing physics."""

    def __init__(self, directory: Path, interval_s: float = .5):
        self.path = Path(directory) / "latest.json"
        self.interval_s = interval_s
        self.last_publish = -float("inf")
        self.disabled = False

    def publish(self, frames):
        now = time.monotonic()
        if self.disabled or now - self.last_publish < self.interval_s:
            return
        try:
            payload = {}
            for view in ("front", "wrist"):
                stream = BytesIO()
                Image.fromarray(frames[view]).save(stream, format="JPEG", quality=80)
                payload[view] = base64.b64encode(stream.getvalue()).decode("ascii")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(json.dumps(payload))
            temporary.replace(self.path)
            self.last_publish = now
        except Exception:
            # A display/cache failure must never change robot execution or evidence.
            self.disabled = True
            get_logger("preview").warning("Live preview unavailable; recording continues", exc_info=True)


def read_preview(directory: Path):
    """Return decoded images so Gradio caches by image content, not a fixed path."""
    try:
        payload = json.loads((Path(directory) / "latest.json").read_text())
        images = []
        for view in ("front", "wrist"):
            with Image.open(BytesIO(base64.b64decode(payload[view], validate=True))) as image:
                images.append(image.convert("RGB"))
        return images
    except (OSError, ValueError, KeyError, TypeError):
        return [None, None]
