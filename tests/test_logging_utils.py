import json
import logging

import pytest

from src.utils.logging_utils import append_json, configure_logging, get_logger, redact_secrets, write_json


@pytest.fixture(autouse=True)
def restore_handlers():
    logger = get_logger()
    before = logger.handlers[:]
    yield
    for handler in list(logger.handlers):
        if handler not in before:
            logger.removeHandler(handler)
            handler.close()


def test_json_records_preserve_bytes_and_prior_events(tmp_path):
    row = {'z': [True, None, 0.5], 'a': 'record ✓'}
    path = tmp_path / 'events.jsonl'
    original = b'{"event": "previous"}\n'
    path.write_bytes(original)
    append_json(path, row)
    assert path.read_bytes() == original + (json.dumps(row, sort_keys=True, allow_nan=False) + '\n').encode()
    write_json(tmp_path / 'result.json', row)
    assert (tmp_path / 'result.json').read_text() == json.dumps(row, sort_keys=True, indent=2, allow_nan=False) + '\n'


def test_runtime_logs_redact_credentials_and_do_not_duplicate(tmp_path, monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'private-test-key')
    log = tmp_path / 'runtime.log'
    configure_logging(log)
    configure_logging(log)
    get_logger('test').info('credential %s', 'private-test-key')
    text = log.read_text()
    assert 'private-test-key' not in text
    assert text.count('credential [redacted]') == 1
    assert redact_secrets('Authorization: Bearer credential') == 'Authorization: Bearer [redacted]'


def test_existing_log_is_not_overwritten_and_next_episode_is_separate(tmp_path):
    first = tmp_path / 'first.log'
    second = tmp_path / 'second.log'
    configure_logging(first)
    get_logger('test').info('first episode')
    snapshot = first.read_bytes()
    configure_logging(second)
    get_logger('test').info('second episode')
    assert first.read_bytes() == snapshot
    assert 'first episode' not in second.read_text()
    with pytest.raises(FileExistsError):
        configure_logging(first)
    assert first.read_bytes() == snapshot
