#!/usr/bin/env python3
"""Download the pinned SAM2 checkpoint atomically, preserving existing files."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def provision(destination, model):
    destination = Path(destination).absolute()
    if {'third_party', 'vendor'} & set(destination.parts):
        raise ValueError('Checkpoint output must be outside third_party and vendor')
    for path in (destination, *destination.parents):
        if path.is_symlink():
            raise ValueError(f'Refusing a symlinked checkpoint path: {path}')
    if destination.exists():
        if digest(destination) != model['sha256']:
            raise ValueError(f'Existing checkpoint has the wrong SHA-256; left unchanged: {destination}')
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=destination.name + '.download.', dir=destination.parent)
    try:
        with os.fdopen(descriptor, 'wb') as output, urlopen(model['url'], timeout=60) as response:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        if digest(temporary) != model['sha256']:
            raise ValueError('Downloaded SAM2 checkpoint failed SHA-256 verification')
        # Linking creates the final name atomically and cannot overwrite a concurrent download.
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if digest(destination) != model['sha256']:
                raise ValueError('Concurrent checkpoint has the wrong SHA-256; left unchanged')
    finally:
        Path(temporary).unlink(missing_ok=True)
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    model = json.loads((ROOT / 'configs/checkpoints.json').read_text())['sam2']
    checkpoint = provision(args.destination, model)
    print(f'Verified SAM2 checkpoint: {checkpoint}')
