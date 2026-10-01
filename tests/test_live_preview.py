"""Live display reuses captured pixels and never changes recording or motion."""
import json
from types import SimpleNamespace

import numpy as np

from src.runtime.live_preview import LivePreview, read_preview
from src.runtime.recording import AgentInterfaceRecorder
from src.ui.runner import current_activity


def test_preview_throttles_and_publishes_both_views_atomically(tmp_path, monkeypatch):
    clock = [10.]
    monkeypatch.setattr("src.runtime.live_preview.time.monotonic", lambda: clock[0])
    preview = LivePreview(tmp_path)
    first = {"front": np.zeros((8, 12, 3), dtype=np.uint8),
             "wrist": np.full((8, 12, 3), 255, dtype=np.uint8)}
    preview.publish(first)
    before = preview.path.read_bytes()
    frames = read_preview(tmp_path)
    assert np.asarray(frames[0]).max() == 0
    assert np.asarray(frames[1]).min() == 255
    preview.publish(dict(front=first["wrist"], wrist=first["front"]))
    assert preview.path.read_bytes() == before
    clock[0] += .5
    preview.publish(dict(front=first["wrist"], wrist=first["front"]))
    assert np.asarray(read_preview(tmp_path)[0]).min() == 255
    assert not preview.path.with_suffix(".tmp").exists()
    assert not np.any(first["front"])


def test_missing_or_incomplete_preview_is_not_an_error(tmp_path):
    assert read_preview(tmp_path) == [None, None]
    (tmp_path / "latest.json").write_text('{"front":')
    assert read_preview(tmp_path) == [None, None]


def test_activity_uses_latest_complete_event_without_changing_log(tmp_path):
    path = tmp_path / "events.jsonl"
    assert current_activity(path) == "Preparing episode"
    path.write_text(json.dumps(dict(kind="agent_input", role="grasp")) + "\n")
    assert current_activity(path) == "Waiting for LLM response · Grasp"
    with path.open("a") as stream:
        stream.write(json.dumps(dict(kind="tool_called", role="grasp", tool="execute_grasp")) + "\n{")
    before = path.read_bytes()
    assert current_activity(path) == "Grasping · Grasp"
    assert path.read_bytes() == before


def record(tmp_path, *, preview=False, broken_preview=False):
    class Environment:
        _sim_step_count = 0
        _control_freq = 20
        _render_width, _render_height = 12, 8
        _record_frames, _subsample_rate = False, 4

        def __init__(self):
            self.renders = []

        def get_simulation_time_s(self):
            return self._sim_step_count / self._control_freq

        def _record_frame(self):
            pass

        def step(self):
            self._sim_step_count += 1
            self._record_frame()

        def render_rgb(self, name):
            self.renders.append((name, self._sim_step_count))
            return np.full((8, 12, 3), self._sim_step_count * 30, dtype=np.uint8)

    class Writer:
        def __init__(self):
            self.frames = []
        def append_data(self, frame):
            self.frames.append(frame.copy())
        def close(self):
            pass

    env, writers = Environment(), []
    connector = SimpleNamespace(env=env, execute_trajectory=lambda *args: None)
    def factory(*args, **kwargs):
        writer = Writer()
        writers.append(writer)
        return writer
    preview_dir = tmp_path / "preview" if preview else None
    if broken_preview:
        tmp_path.mkdir(parents=True)
        preview_dir.write_text("not a directory")
    recorder = AgentInterfaceRecorder(connector=connector, ee_connector=connector,
        output_dir=tmp_path / "interface", objective="test", model="no-llm",
        writer_factory=factory, preview_dir=preview_dir)
    cameras = {name: dict(intrinsics=np.eye(3), camera_to_base={})
               for name in ("agentview", "robot0_eye_in_hand")}
    recorder._robot_camera_snapshot = lambda: (cameras, dict(position=dict(x=0., y=0., z=0.)))
    if recorder.preview:
        recorder.preview.interval_s = 0
    recorder.install()
    recorder.capture(force=True)
    assert env._sim_step_count == 0 and recorder.frame_count == 1
    for _ in range(6):
        env.step()
    recorder.close()
    rows = [json.loads(line) for line in (recorder.output_dir / "frames.jsonl").read_text().splitlines()]
    assert [row["simulator_step"] for row in rows] == [0, 2, 4, 6]
    assert env._sim_step_count == 6 and recorder.error is None
    return env.renders, [np.stack(writer.frames) for writer in writers]


def test_preview_preserves_recorded_pixels_render_count_and_physics(tmp_path):
    baseline = record(tmp_path / "baseline")
    live = record(tmp_path / "live", preview=True)
    assert baseline[0] == live[0]
    for before, after in zip(baseline[1], live[1]):
        np.testing.assert_array_equal(before, after)
    assert all(image is not None for image in read_preview(tmp_path / "live/preview"))


def test_preview_write_failure_cannot_abort_recording(tmp_path):
    record(tmp_path / "broken", preview=True, broken_preview=True)
