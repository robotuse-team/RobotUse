"""Launch the existing episode command and read its independent verifier."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import signal
import subprocess
import threading
from datetime import datetime, timezone
from uuid import uuid4

from src.runtime.paths import REPOSITORY_ROOT
from src.agent.playbook import VERSION_CHOICES
from src.runtime.live_preview import read_preview
from src.utils.logging_utils import get_logger, redact_secrets

LOGGER = get_logger(__name__)
METADATA = Path("src/simulator/robolab/third_party/robolab/robolab/tasks/_metadata/task_metadata.json")


def display_text(value: str) -> str:
    """Redact credentials for public presentation without modifying artifacts."""
    return redact_secrets(value)


def task_catalog(root: Path = REPOSITORY_ROOT) -> dict[str, dict]:
    """Read upstream metadata without importing Isaac Sim or task implementations."""
    source = root / METADATA
    if not source.is_file():
        raise RuntimeError("RoboLab task metadata is missing. Run scripts/setup/sources.sh first.")
    return {row["task_name"]: row for row in json.loads(source.read_text())}


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def verifier_status(data: dict) -> str:
    if data.get("evaluated") is not True:
        verdict = "Not evaluated"
    elif data.get("task_success") is True:
        verdict = "Passed"
    elif data.get("task_success") is False:
        verdict = "Failed"
    else:
        verdict = "Unknown"
    score = data.get('score')
    if (data.get('score_evaluated') is True and type(score) in (int, float)
            and math.isfinite(score) and 0 <= score <= 1):
        return f'{verdict} · Score {score:.3f}'
    if data.get('score_enabled') is True:
        return f'{verdict} · Score unavailable'
    if data.get('score_enabled') is False:
        return f'{verdict} · Score off'
    return verdict


def current_activity(path: Path) -> str:
    """Read the last complete event while the episode continues appending."""
    try:
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - 65536))
            lines = stream.read().splitlines()
    except OSError:
        return "Preparing episode"
    for line in reversed(lines):
        try:
            event = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            continue
        if not isinstance(event, dict):
            continue
        kind, role = event.get("kind"), str(event.get("role", "agent")).capitalize()
        if kind == "agent_input":
            return f"Waiting for LLM response · {role}"
        if kind == "tool_called":
            tool = str(event.get("tool", "tool"))
            labels = {"select_region": "Selecting object", "grasp_candidates": "Planning grasp",
                      "execute_grasp": "Grasping", "execute_place": "Placing",
                      "release": "Releasing", "move": "Moving", "observe": "Observing"}
            return f"{labels.get(tool, tool.replace('_', ' ').capitalize())} · {role}"
        if kind in ("session_error", "tool_error"):
            return f"Handling an error · {role}"
        if kind == "tool_result":
            return f"Processing tool result · {role}"
        if kind == "session_closed":
            return f"Agent finished · {role}"
    return "Preparing episode"


class EpisodeRunner:
    """One simulator process at a time; each launch gets a new artifact directory."""

    def __init__(self, root: Path = REPOSITORY_ROOT, runs_root: Path | None = None):
        self.root = root.resolve()
        self.runs_root = (runs_root or self.root / "runs/ui").resolve()
        self.catalog = task_catalog(self.root)
        self.lock = threading.Lock()
        self.process: subprocess.Popen | None = None
        self.active: Path | None = None

    def start(self, task: str, seed: int, playbook: str, dry_run: bool) -> str:
        if task not in self.catalog:
            raise ValueError("Select a task from the catalog.")
        if isinstance(seed, bool) or int(seed) != seed or not 0 <= seed <= 2**32 - 1:
            raise ValueError("Seed must be an integer between 0 and 4294967295.")
        if playbook not in VERSION_CHOICES:
            raise ValueError("Select a playbook version.")
        with self.lock:
            if self.active is not None:
                raise RuntimeError("An episode is already running. Stop it or wait for completion.")
            episode_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid4().hex[:8]
            run = self.runs_root / episode_id
            run.mkdir(parents=True, exist_ok=False)
            env = os.environ.copy()
            env["ROBOTUSE_LIVE_PREVIEW_DIR"] = str(run / "preview")
            if dry_run:
                runtime = Path(env.get("ROBOTUSE_RUNTIME_ROOT", self.root / "runtime"))
                environment = Path(env.get("CPU_ENVIRONMENT") or runtime / "tool-envs/cpu")
                env["ROBOLAB_PYTHON"] = env.get("ROBOTUSE_CPU_PYTHON") or str(environment / "bin/python")
            command = [str(self.root / "scripts/run/robolab.sh"), "--task", task,
                       "--seed", str(int(seed)), "--difficulty", "auto", "--playbook-version", playbook,
                       "--output-dir", str(run / "episode")]
            if dry_run:
                command.append("--dry-run")
            request = dict(task=task, seed=int(seed), playbook=playbook, dry_run=bool(dry_run),
                           started_at=datetime.now(timezone.utc).isoformat())
            with (run / "request.json").open("x") as stream:
                json.dump(request, stream, indent=2)
            try:
                process = subprocess.Popen(command, cwd=self.root, env=env, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True, errors="replace",
                                           start_new_session=True, bufsize=1)
            except OSError as exc:
                self._finish(run, None, str(exc))
                return episode_id
            self.process, self.active = process, run
            threading.Thread(target=self._collect, args=(run, process), daemon=True).start()
            LOGGER.info("Episode %s started: %s, seed %s, configuration check=%s", episode_id, task, seed, dry_run)
            return episode_id

    def _finish(self, run: Path, returncode: int | None, error: str | None = None):
        with (run / "launcher.json").open("x") as stream:
            json.dump(dict(returncode=returncode, error=redact_secrets(error) if error else None,
                           finished_at=datetime.now(timezone.utc).isoformat()), stream, indent=2)

    def _collect(self, run: Path, process: subprocess.Popen):
        error = None
        try:
            with (run / "controller.log").open("x") as log:
                for line in process.stdout:
                    log.write(redact_secrets(line))
                    log.flush()
            process.wait()
        except Exception as exc:
            error = str(exc)
            self._terminate(process)
        finally:
            process.stdout.close()
            with self.lock:
                try:
                    self._finish(run, process.returncode, error)
                except OSError:
                    LOGGER.exception("Could not write the final process record for %s", run.name)
                finally:
                    self.process, self.active = None, None
            LOGGER.info("Episode %s process ended with code %s", run.name, process.returncode)

    @staticmethod
    def _terminate(process: subprocess.Popen):
        try:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            # A worker can outlive its parent after SIGTERM; reap the entire group.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)
        except ProcessLookupError:
            pass

    def stop(self, episode_id: str):
        with self.lock:
            if self.active is None or self.active.name != episode_id:
                return
            process = self.process
            marker = self.active / "cancel_requested.json"
            if marker.exists():
                return
            with marker.open("x") as stream:
                json.dump({"requested_at": datetime.now(timezone.utc).isoformat()}, stream)
            threading.Thread(target=self._terminate, args=(process,), daemon=True).start()

    def close(self):
        with self.lock:
            process = self.process
        if process is not None:
            self._terminate(process)

    def episodes(self) -> list[str]:
        if not self.runs_root.exists():
            return []
        return sorted((path.name for path in self.runs_root.iterdir()
                       if path.is_dir() and not path.is_symlink() and (path / "request.json").is_file()), reverse=True)[:100]

    def history(self, episode_id, turn_id=None, *, follow_latest=False, offset=0):
        from src.ui.history import read_history
        episode = self.runs_root / episode_id / "episode" if episode_id in self.episodes() else None
        return read_history(episode, turn_id, follow_latest=follow_latest, offset=offset)

    def snapshot(self, episode_id: str | None) -> dict:
        if not episode_id or episode_id not in self.episodes():
            return dict(status="Ready", verifier="Not evaluated", logs="", details={},
                        videos=[None, None], previews=[None, None], activity="")
        run = self.runs_root / episode_id
        request, launcher = read_json(run / "request.json"), read_json(run / "launcher.json")
        verdict = read_json(run / "episode/verifier.json")
        cancelled = (run / "cancel_requested.json").exists()
        if launcher:
            status = f"Process completed · exit {launcher['returncode']}"
            if cancelled:
                status = f"Stopped · exit {launcher['returncode']}"
            elif request.get("dry_run") and launcher["returncode"] == 0:
                status = "Configuration valid"
            elif launcher.get("error"):
                status = "Could not start · " + launcher["error"]
        else:
            with self.lock:
                active_here = self.active == run
            status = ("Stopping…" if cancelled else "Running") if active_here else "Interrupted · no final process record"
        log_path = run / "controller.log"
        logs = ""
        if log_path.exists():
            with log_path.open("rb") as stream:
                stream.seek(max(0, log_path.stat().st_size - 24000))
                logs = redact_secrets(stream.read().decode("utf-8", errors="replace"))
        if request.get("dry_run") and launcher.get("returncode") == 0:
            logs = "Configuration validation completed. No simulator, checkpoint, or LLM was started."
        recording_complete = read_json(run / "episode/interface/manifest.json").get("status") == "completed"
        videos = [str(run / f"episode/interface/{view}.mp4")
                  if launcher and recording_complete and (run / f"episode/interface/{view}.mp4").is_file() else None
                  for view in ("front", "wrist")]
        details = dict(task=request.get("task"), seed=request.get("seed"), playbook=request.get("playbook"),
                       artifacts=str(run), verifier_source=str(run / "episode/verifier.json") if verdict else None,
                       verifier=verdict)
        activity = (current_activity(run / "episode/events.jsonl")
                    if not launcher and active_here and not cancelled else status)
        return dict(status=status, verifier=verifier_status(verdict), logs=logs, details=details,
                    videos=videos, previews=read_preview(run / "preview"), activity=activity)
