"""Exercise real process lifecycle without importing a simulator or web framework."""
import json
import sys
import time

import pytest

from src.ui.runner import EpisodeRunner, METADATA, display_text, verifier_status


@pytest.fixture
def runner(tmp_path):
    metadata = tmp_path / METADATA
    metadata.parent.mkdir(parents=True)
    metadata.write_text(json.dumps([
        dict(task_name=task, instruction=task, difficulty_label="simple", episode_s="30")
        for task in ["PassTask", "FailTask", "LongTask"]]))
    script = tmp_path / "scripts/run/robolab.sh"
    script.parent.mkdir(parents=True)
    script.write_text(f"#!{sys.executable}\n" + '''
import json,os,pathlib,subprocess,sys,time
args=sys.argv[1:]
output=pathlib.Path(args[args.index('--output-dir')+1])
task=args[args.index('--task')+1]
print('credential='+os.environ.get('GOOGLE_API_KEY','none'), flush=True)
if '--dry-run' in args:
    print(json.dumps({'configuration':'RobotUse','task':task}))
    raise SystemExit(0)
if task=='LongTask':
    subprocess.Popen([sys.executable, '-c', 'import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(30)'])
    time.sleep(30)
output.mkdir()
(output/'verifier.json').write_text(json.dumps({'evaluated':True,'task_success':task=='PassTask','reward':int(task=='PassTask')}))
raise SystemExit(1 if task=='PassTask' else 0)
''')
    script.chmod(0o755)
    result = EpisodeRunner(tmp_path)
    yield result
    result.close()


def wait_done(runner, episode):
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        state = runner.snapshot(episode)
        if state["status"].startswith(("Process completed", "Stopped", "Configuration valid", "Could not start")):
            return state
        time.sleep(.03)
    pytest.fail("Episode did not finish")


@pytest.mark.parametrize(("task", "verdict", "exit_code"), [("PassTask", "Passed", 1), ("FailTask", "Failed", 0)])
def test_verifier_is_independent_of_process_exit(runner, task, verdict, exit_code):
    episode = runner.start(task, 4, "v0", False)
    state = wait_done(runner, episode)
    assert state["verifier"] == verdict
    assert f"exit {exit_code}" in state["status"]
    assert state["details"]["verifier_source"].endswith("episode/verifier.json")


def test_dry_run_is_not_task_success_and_records_do_not_overwrite(runner, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "secret-test-value")
    first = runner.start("PassTask", 0, "v0", True)
    state = wait_done(runner, first)
    assert state["verifier"] == "Not evaluated"
    assert state["status"].startswith("Configuration valid")
    assert "secret-test-value" not in state["logs"]
    recorded = {path: path.read_bytes() for path in (runner.runs_root / first).iterdir() if path.is_file()}
    assert b"secret-test-value" not in recorded[runner.runs_root / first / "controller.log"]
    second = runner.start("PassTask", 0, "v0", True)
    wait_done(runner, second)
    assert first != second
    assert all(path.read_bytes() == data for path, data in recorded.items())


def test_single_active_process_and_cancel(runner):
    episode = runner.start("LongTask", 0, "v0", False)
    with pytest.raises(RuntimeError, match="already running"):
        runner.start("PassTask", 0, "v0", False)
    time.sleep(.1)
    runner.stop(episode)
    state = wait_done(runner, episode)
    assert state["status"].startswith("Stopped")
    assert state["verifier"] == "Not evaluated"
    assert runner.process is None


@pytest.mark.parametrize(("task", "seed", "playbook"), [
    ("--arbitrary-shell", 0, "v0"), ("PassTask", -1, "v0"),
    ("PassTask", .5, "v0"), ("PassTask", 0, "invalid")])
def test_invalid_settings_never_launch(runner, task, seed, playbook):
    with pytest.raises(ValueError):
        runner.start(task, seed, playbook, False)
    assert runner.episodes() == []


def test_missing_or_malformed_verifier_cannot_claim_success(runner):
    assert verifier_status({}) == "Not evaluated"
    assert verifier_status(dict(evaluated=False, task_success=True)) == "Not evaluated"
    assert verifier_status(dict(evaluated=True, task_success="true")) == "Unknown"
    assert runner.snapshot("../outside")["status"] == "Ready"


def test_final_record_write_failure_releases_process_slot(runner, monkeypatch):
    def fail_write(*args, **kwargs):
        raise OSError("disk full")
    monkeypatch.setattr(runner, "_finish", fail_write)
    episode = runner.start("PassTask", 0, "v0", True)
    deadline = time.monotonic() + 5
    while runner.active is not None and time.monotonic() < deadline:
        time.sleep(.03)
    assert runner.active is None
    assert runner.process is None
    assert runner.snapshot(episode)["status"].startswith("Interrupted")


def test_public_log_rendering_redacts_secrets_without_mutating_record(tmp_path, monkeypatch):
    artifact = tmp_path / "controller.log"
    monkeypatch.setenv("GOOGLE_API_KEY", "secret-test-value")
    raw = 'ROBOTUSE_CONFIGURATION {"configuration": "RobotUse", "credential": "secret-test-value"}\n'
    artifact.write_text(raw)
    displayed = display_text(artifact.read_text())
    assert displayed.startswith("ROBOTUSE_CONFIGURATION")
    assert "secret-test-value" not in displayed
    assert '"configuration": "RobotUse"' in displayed
    assert artifact.read_text() == raw


def test_live_snapshot_updates_before_process_finishes(runner):
    import numpy as np
    from src.runtime.live_preview import LivePreview

    episode = runner.start("LongTask", 0, "v0", False)
    run = runner.runs_root / episode
    output = run / "episode"
    output.mkdir()
    event = dict(kind="agent_input", role="prime")
    (output / "events.jsonl").write_text(json.dumps(event) + "\n")
    preview = LivePreview(run / "preview", interval_s=0)
    preview.publish({view: np.zeros((8, 12, 3), dtype=np.uint8) for view in ("front", "wrist")})
    state = runner.snapshot(episode)
    assert state["status"] == "Running"
    assert state["activity"] == "Waiting for LLM response · Prime"
    assert all(image is not None for image in state["previews"])
    assert state["videos"] == [None, None]
    assert state["verifier"] == "Not evaluated"
    preview.publish({view: np.full((8, 12, 3), 255, dtype=np.uint8) for view in ("front", "wrist")})
    assert np.asarray(runner.snapshot(episode)["previews"][0]).min() == 255
    runner.stop(episode)
    state = wait_done(runner, episode)
    assert state["activity"].startswith("Stopped")
    assert state["verifier"] == "Not evaluated"


def test_stopped_partial_video_keeps_preview_and_history_readable(runner):
    episode = runner.start("LongTask", 0, "v0", False)
    run = runner.runs_root / episode
    interface = run / "episode/interface"
    interface.mkdir(parents=True)
    (interface / "front.mp4").write_bytes(b"unfinished mp4")
    (interface / "manifest.json").write_text(json.dumps({"status": "recording"}))
    (run / "episode/events.jsonl").write_text(json.dumps(dict(
        kind="agent_input", seq=0, session_id="p", role="prime", instruction="Pick banana")) + "\n")
    runner.stop(episode)
    state = wait_done(runner, episode)
    assert state["videos"] == [None, None]
    assert runner.history(episode)["input"]["instruction"] == "Pick banana"
    assert runner.history("../outside")["count"] == 0
