"""Length-prefixed msgpack framing for the stdio tool RPC.

Each frame is a single msgpack object preceded by a 4-byte big-endian
length header. Numpy arrays serialize transparently via ``msgpack_numpy``
so tool signatures that take/return ``np.ndarray`` work without bespoke
wire formats. Other gap_core.types values (Se3Pose, Mask, …) are
``TypedDict``s — they round-trip as plain dicts.

Frame shapes (kept loose — see ``protocol.py`` for the type-checked
helpers, but the wire itself is just ``dict[str, Any]``):

    {"id": "req-1", "kind": "call", "tool": "sam3.segment", "args": {...}}
    {"id": "req-1", "kind": "result", "result": {...}}
    {"id": "req-1", "kind": "error", "error": {"type": "...", "message": "..."}}
    {"id": "hs-0", "kind": "catalog", "tools": [{...}, ...]}
"""

from __future__ import annotations

import struct
from typing import IO, Any

import msgpack
import msgpack_numpy as _msgpack_numpy

# Pack numpy arrays as `{"nd": True, "type": "<f4", "shape": [...], "data": b"..."}`.
# Patching at module load means every msgpack.packb / msgpack.unpackb in this
# process supports ndarray out of the box.
_msgpack_numpy.patch()

_LEN_HEADER = struct.Struct(">I")  # 4-byte big-endian unsigned length


class FrameError(RuntimeError):
    """Raised when a frame can't be decoded (short read, malformed payload)."""


def encode_frame(payload: dict[str, Any]) -> bytes:
    """Pack ``payload`` as ``<4-byte length><msgpack bytes>``."""
    body = msgpack.packb(payload, use_bin_type=True)
    assert body is not None  # msgpack returns bytes for non-None input
    return _LEN_HEADER.pack(len(body)) + body


def decode_frame(stream: IO[bytes]) -> dict[str, Any] | None:
    """Read one frame off ``stream``; return ``None`` at clean EOF.

    Raises :class:`FrameError` on partial reads (stream closed mid-frame)
    or unpack failures. The caller is responsible for distinguishing EOF
    (``None``) from error (raised) — EOF on a clean boundary is normal
    teardown.
    """
    header = stream.read(_LEN_HEADER.size)
    if not header:
        return None
    if len(header) != _LEN_HEADER.size:
        raise FrameError(
            f"short read on length header: got {len(header)} bytes, "
            f"expected {_LEN_HEADER.size}"
        )
    (length,) = _LEN_HEADER.unpack(header)
    body = stream.read(length)
    if len(body) != length:
        raise FrameError(
            f"short read on body: got {len(body)} bytes, expected {length}"
        )
    try:
        obj = msgpack.unpackb(body, raw=False)
    except Exception as exc:
        raise FrameError(f"msgpack decode failed: {exc}") from exc
    if not isinstance(obj, dict):
        raise FrameError(
            f"frame payload must be a mapping, got {type(obj).__name__}"
        )
    return obj


def write_frame(stream: IO[bytes], payload: dict[str, Any]) -> None:
    """Encode + write + flush. The flush is what makes the RPC interactive."""
    stream.write(encode_frame(payload))
    stream.flush()


__all__ = ["FrameError", "decode_frame", "encode_frame", "write_frame"]
