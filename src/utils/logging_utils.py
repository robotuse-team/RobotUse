"""Application logging; structured execution records keep their original format."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import re
import threading

_LOCK = threading.RLock()
_CREDENTIAL_NAMES = ('OPENROUTER_API_KEY', 'GOOGLE_AI_STUDIO_KEY', 'GOOGLE_API_KEY',
                     'GEMINI_API_KEY', 'OPENAI_API_KEY', 'ANTHROPIC_API_KEY')


def redact_secrets(text: str) -> str:
    """Remove known credentials from human-readable logs, including tracebacks."""
    for name in _CREDENTIAL_NAMES:
        value = os.environ.get(name)
        if value:
            text = text.replace(value, '[redacted]')
    text = re.sub(r'(?i)(authorization\s*[:=]\s*[\"\x27]?bearer\s+)\S+', r'\1[redacted]', text)
    return text


def redact_data(value):
    """Copy structured display data while redacting strings before serialization."""
    if isinstance(value, str):
        return redact_secrets(value)
    if isinstance(value, dict):
        return {redact_secrets(key) if isinstance(key, str) else key: redact_data(item)
                for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_data(item) for item in value]
    return value


class _Formatter(logging.Formatter):
    def format(self, record):
        return redact_secrets(super().format(record))


def get_logger(name: str = '') -> logging.Logger:
    return logging.getLogger('robotuse' + ('.' + name if name else ''))


def configure_logging(path=None, *, level=logging.INFO) -> logging.Logger:
    """Configure only our namespace; refuse to overwrite an existing log file."""
    logger = get_logger()
    with _LOCK:
        logger.setLevel(level)
        logger.propagate = False
        formatter = _Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s')
        if not any(getattr(handler, '_robotuse_console', False) for handler in logger.handlers):
            handler = logging.StreamHandler()
            handler._robotuse_console = True
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        if path is not None:
            destination = Path(path).absolute()
            if not any(getattr(handler, 'baseFilename', None) == str(destination) for handler in logger.handlers):
                handler = logging.FileHandler(destination, mode='x', encoding='utf-8')
                handler.setFormatter(formatter)
                for previous in list(logger.handlers):
                    if isinstance(previous, logging.FileHandler):
                        logger.removeHandler(previous)
                        previous.close()
                logger.addHandler(handler)
    return logger


def write_json(path, data):
    """Preserve the existing JSON artifact serialization contract."""
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + '\n')


def append_json(path, data):
    """Append a record without rewriting preceding execution evidence."""
    with Path(path).open('a') as stream:
        stream.write(json.dumps(data, sort_keys=True, allow_nan=False) + '\n')
