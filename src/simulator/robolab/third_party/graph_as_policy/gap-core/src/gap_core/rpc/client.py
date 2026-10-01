"""gap-side ToolClient — drives one persistent bundle subprocess over stdio.

Spawns ``uv run --project <bundle_dir> -- python -m gap_core.rpc.server
--bundle <name>``, runs the catalog handshake to discover the bundle's
exported tools, and dispatches subsequent calls by writing ``call`` frames
to the subprocess's stdin and reading ``result``/``error`` frames back.

One ToolClient per bundle per workflow execution. The client is sequential:
one in-flight call at a time per bundle (a tool that needs parallelism gets
its own server process; this is out of scope for v1). Thread-safety is
enforced by a single lock around the request/response pair.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import threading
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .codec import FrameError, decode_frame, write_frame

logger = logging.getLogger(__name__)


#: Default per-call reply timeout (seconds). A single tool call that takes
#: longer than this is almost always a hang (e.g. a deadlocked subprocess),
#: not legitimate work — perception and planning calls finish in seconds to
#: low minutes. Override with ``GAP_TOOL_CALL_TIMEOUT_S`` (set to ``0`` to
#: disable) or the ``call_timeout_s`` constructor arg.
DEFAULT_TOOL_CALL_TIMEOUT_S = 600.0


def _resolve_call_timeout(explicit: float | None) -> float | None:
    """Resolve the per-call reply timeout in seconds, or ``None`` to disable.

    Precedence: explicit arg > ``GAP_TOOL_CALL_TIMEOUT_S`` env > default.
    A non-positive value disables the timeout.
    """
    if explicit is not None:
        return explicit if explicit > 0 else None
    raw = os.environ.get("GAP_TOOL_CALL_TIMEOUT_S", "").strip()
    if raw:
        try:
            v = float(raw)
        except ValueError:
            return DEFAULT_TOOL_CALL_TIMEOUT_S
        return v if v > 0 else None
    return DEFAULT_TOOL_CALL_TIMEOUT_S


class ToolClientError(RuntimeError):
    """The bundle server died, mis-framed, or returned an unparseable result."""


class ToolCallTimeout(ToolClientError):
    """A tool call exceeded its reply timeout; the subprocess was terminated.

    Subclass of :class:`ToolClientError` so existing protocol-failure handling
    (a node raises → the subgraph's ``on_error`` exit fires) catches it, while
    callers that care can still distinguish a hang from a crash.
    """


class ToolRemoteError(RuntimeError):
    """The bundle server reported a tool-side exception via the ``error`` frame.

    Surfaces the remote exception type and (when present) the remote
    traceback so the gap-side stack trace points at the actual failure
    site in the bundle's tools.py.
    """

    def __init__(self, tool: str, remote_type: str, message: str, remote_tb: str = ""):
        super().__init__(f"{tool}: {remote_type}: {message}")
        self.tool = tool
        self.remote_type = remote_type
        self.remote_tb = remote_tb


@dataclass
class CatalogEntry:
    """One tool's metadata as reported by the server's startup handshake."""

    name: str
    summary: str = ""
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_inputs: dict[str, dict[str, Any]] = field(default_factory=dict)
    schema_outputs: dict[str, dict[str, Any]] = field(default_factory=dict)


def _terminate_group(proc: subprocess.Popen, grace_s: float) -> None:
    """SIGTERM the process group, wait, then SIGKILL.

    Mirrors gap.runtime.policy_manager._terminate_group so both bundle
    flavors (policies, tools) tear down their subprocesses the same way.
    """
    if proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=max(0.1, grace_s))
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        logger.warning(
            "tool bundle subprocess pid=%d did not exit after SIGKILL",
            proc.pid,
        )


class ToolClient:
    """A long-lived stdio RPC channel to one bundle's tool server."""

    def __init__(
        self,
        bundle_name: str,
        bundle_dir: str | Path,
        *,
        command: Iterable[str] | None = None,
        env: dict[str, str] | None = None,
        evict_grace_s: float = 5.0,
        call_timeout_s: float | None = None,
        use_uv: bool = True,
    ) -> None:
        self.bundle_name = bundle_name
        self.bundle_dir = Path(bundle_dir)
        self._grace = float(evict_grace_s)
        self._call_timeout_s = _resolve_call_timeout(call_timeout_s)
        self._lock = threading.Lock()
        self._req_counter = 0

        cmd = list(command) if command else [
            "python", "-m", "gap_core.rpc.server", "--bundle", bundle_name,
        ]
        # Default: run the server inside the bundle's own venv via uv. Tests
        # (and callers that already resolved an interpreter) pass use_uv=False
        # to spawn `cmd` directly.
        argv = (["uv", "run", "--project", str(self.bundle_dir), "--", *cmd]
                if use_uv else list(cmd))
        full_env = {**os.environ, **(env or {})}
        logger.info("[tool-bundle:%s] spawning: %s", bundle_name, " ".join(argv))
        self._proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,  # let the bundle's stderr flow to gap's stderr
            cwd=str(self.bundle_dir),
            env=full_env,
            start_new_session=True,  # own process group for clean teardown
        )

        try:
            self._catalog = self._read_catalog()
        except BaseException:
            self.close()
            raise

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def catalog(self) -> list[CatalogEntry]:
        """List of tools the bundle server announced at startup."""
        return list(self._catalog)

    def call(self, tool: str, /, **kwargs: Any) -> Any:
        """Dispatch ``tool(**kwargs)`` to the bundle server.

        Raises:
            ToolRemoteError: tool raised in the bundle (carries the remote
                exception type + message + traceback).
            ToolClientError: protocol-level failure (subprocess died,
                mis-framed reply).
        """
        with self._lock:
            self._req_counter += 1
            rid = f"req-{self._req_counter}"
            self._write({"id": rid, "kind": "call", "tool": tool, "args": kwargs})

            if not self._call_timeout_s:
                return self._read_reply(rid)

            # Bound the blocking reply read with a watchdog. The read sits in
            # a C-level pipe read that a cooperative flag can't interrupt, so
            # on expiry we terminate the subprocess: closing its stdout makes
            # the read return EOF, which unblocks _read_reply (it then raises
            # ToolClientError, which we translate to ToolCallTimeout).
            timed_out = threading.Event()

            def _fire() -> None:
                timed_out.set()
                logger.warning(
                    "[tool-bundle:%s] tool %r exceeded %.0fs timeout; "
                    "terminating subprocess", self.bundle_name, tool,
                    self._call_timeout_s,
                )
                _terminate_group(self._proc, self._grace)

            timer = threading.Timer(self._call_timeout_s, _fire)
            timer.daemon = True
            timer.start()
            try:
                return self._read_reply(rid)
            except ToolClientError:
                if timed_out.is_set():
                    raise ToolCallTimeout(
                        f"[tool-bundle:{self.bundle_name}] tool {tool!r} "
                        f"exceeded {self._call_timeout_s:.0f}s timeout; "
                        f"subprocess terminated"
                    ) from None
                raise
            finally:
                timer.cancel()

    def close(self) -> None:
        """Terminate the subprocess. Idempotent."""
        if self._proc is None:
            return
        # Close stdin so the server exits its read loop cleanly.
        try:
            if self._proc.stdin and not self._proc.stdin.closed:
                self._proc.stdin.close()
        except Exception:
            pass
        try:
            self._proc.wait(timeout=max(0.5, self._grace))
        except subprocess.TimeoutExpired:
            _terminate_group(self._proc, self._grace)
        finally:
            try:
                if self._proc.stdout and not self._proc.stdout.closed:
                    self._proc.stdout.close()
            except Exception:
                pass

    def __enter__(self) -> ToolClient:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _write(self, payload: dict[str, Any]) -> None:
        stdin = self._proc.stdin
        if stdin is None or stdin.closed:
            raise ToolClientError(
                f"[tool-bundle:{self.bundle_name}] cannot write — stdin closed"
            )
        try:
            write_frame(stdin, payload)
        except BrokenPipeError as exc:
            self._raise_dead("write", exc)

    def _read_reply(self, expected_id: str) -> Any:
        # The server is strictly sequential: one request → one reply. The
        # reply id must match the request id (a sanity check, not a feature
        # — request multiplexing would require a different protocol).
        frame = self._read_frame_or_die()
        if str(frame.get("id", "")) != expected_id:
            raise ToolClientError(
                f"[tool-bundle:{self.bundle_name}] reply id mismatch: "
                f"sent {expected_id!r}, got {frame.get('id')!r}"
            )
        kind = frame.get("kind")
        if kind == "result":
            return frame.get("result")
        if kind == "error":
            err = frame.get("error") or {}
            raise ToolRemoteError(
                tool=str(err.get("tool", "")),
                remote_type=str(err.get("type", "Exception")),
                message=str(err.get("message", "")),
                remote_tb=str(err.get("traceback", "")),
            )
        raise ToolClientError(
            f"[tool-bundle:{self.bundle_name}] unexpected reply kind {kind!r}"
        )

    def _read_catalog(self) -> list[CatalogEntry]:
        frame = self._read_frame_or_die()
        if frame.get("kind") != "catalog":
            raise ToolClientError(
                f"[tool-bundle:{self.bundle_name}] expected catalog handshake "
                f"first, got {frame.get('kind')!r}"
            )
        entries: list[CatalogEntry] = []
        for raw in frame.get("tools") or []:
            schema = raw.get("schema") or {}
            entries.append(CatalogEntry(
                name=str(raw.get("name", "")),
                summary=str(raw.get("summary", "")),
                tags=list(raw.get("tags") or []),
                metadata=dict(raw.get("metadata") or {}),
                schema_inputs=dict(schema.get("inputs") or {}),
                schema_outputs=dict(schema.get("outputs") or {}),
            ))
        return entries

    def _read_frame_or_die(self) -> dict[str, Any]:
        stdout = self._proc.stdout
        if stdout is None:
            self._raise_dead("read", None)
        try:
            frame = decode_frame(stdout)
        except FrameError as exc:
            self._raise_dead("frame", exc)
        if frame is None:
            self._raise_dead("eof", None)
        return frame  # type: ignore[return-value]

    def _raise_dead(self, where: str, exc: BaseException | None) -> "ToolClient":
        rc = self._proc.poll()
        detail = f"({type(exc).__name__}: {exc})" if exc else ""
        raise ToolClientError(
            f"[tool-bundle:{self.bundle_name}] subprocess unreachable on "
            f"{where}: returncode={rc} {detail}".rstrip()
        )


__all__ = [
    "CatalogEntry",
    "DEFAULT_TOOL_CALL_TIMEOUT_S",
    "ToolCallTimeout",
    "ToolClient",
    "ToolClientError",
    "ToolRemoteError",
]
