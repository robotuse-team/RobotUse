"""Recorded turns remain attributable across delegation, partial writes and navigation."""
import hashlib
import json

from PIL import Image

from src.ui.history import read_history


def append(path, row):
    with path.open("a") as stream:
        stream.write(json.dumps(row) + "\n")


def recorded_episode(tmp_path):
    root = tmp_path / "episode"
    root.mkdir()
    for name in ("front", "wrist", "overlay"):
        Image.new("RGB", (24, 16), "red").save(root / f"{name}.png")
    events = [
        dict(seq=0, kind="agent_input", session_id="prime", role="prime", step=0, instruction="Pick banana"),
        dict(seq=1, kind="tool_called", session_id="prime", tool="delegate_point"),
        dict(seq=2, kind="agent_input", session_id="point", role="point", step=0, instruction="Select banana"),
        dict(seq=3, kind="tool_result", session_id="point", tool="select_region", result={"image_refs": ["img_overlay"]}),
        dict(seq=4, kind="tool_result", session_id="prime", tool="delegate_point", result={"point_ref": "banana"}),
        dict(seq=5, kind="agent_input", session_id="prime", role="prime", step=1, instruction="Pick banana"),
    ]
    for event in events:
        append(root / "events.jsonl", event)
    for sid, rid, refs in [("prime", "p1", ["img_front"]), ("point", "q1", ["img_wrist"]),
                           ("prime", "p2", ["img_front", "img_overlay"])]:
        append(root / "provider_inputs.jsonl", dict(session_id=sid, role=sid, request_id=rid,
            time_unix_s=1790841500, attached_image_refs=refs,
            image_paths={"img_front": str(root / "front.png"), "img_wrist": str(root / "wrist.png"),
                         "img_overlay": str(root / "overlay.png")},
            messages=[{"role": "user", "text": [json.dumps({"instruction": rid})]}]))
    # Deliberately shuffled: join by request ID, not response row order.
    for rid, tool in [("q1", "select_region"), ("p1", "delegate_point")]:
        append(root / "provider_audit.jsonl", dict(request_id=rid, response=dict(tool=tool, arguments={})))
    return root


def test_parent_and_child_outputs_do_not_mix(tmp_path):
    root = recorded_episode(tmp_path)
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()}
    parent = read_history(root, "event:0")
    child = read_history(root, "event:2")
    assert parent["count"] == 3
    assert parent["output"]["tool"] == "delegate_point"
    assert [r["tool"] for r in parent["results"]] == ["delegate_point"]
    assert child["output"]["tool"] == "select_region"
    assert [r["tool"] for r in child["results"]] == ["select_region"]
    assert child["input_images"][0][0].endswith("wrist.png")
    assert len(child["input_images"]) == 1  # Only images actually attached to this request.
    assert child["result_images"][0][0].endswith("overlay.png")
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == digest for p, digest in before.items())


def test_navigation_and_new_turns_keep_manual_selection(tmp_path):
    root = recorded_episode(tmp_path)
    assert read_history(root)["selected"] == "event:5"
    assert read_history(root, "event:0", offset=-1)["selected"] == "event:0"
    assert read_history(root, "event:2", offset=1)["selected"] == "event:5"
    assert read_history(root, "event:5", offset=1)["selected"] == "event:5"
    append(root / "events.jsonl", dict(seq=6, kind="agent_input", session_id="grasp", role="grasp", step=0))
    assert read_history(root, "event:0")["selected"] == "event:0"
    latest = read_history(root, "event:0", follow_latest=True)
    assert latest["selected"] == "event:6" and latest["count"] == 4
    assert latest["output"] == {} and "No response recorded" in latest["summary"]


def test_partial_response_is_not_misattributed(tmp_path):
    root = recorded_episode(tmp_path)
    with (root / "provider_audit.jsonl").open("a") as stream:
        stream.write('{"request_id":"p2"')
    pending = read_history(root, "event:5")
    assert pending["output"] == {}
    assert "No response recorded" in pending["summary"]
    with (root / "provider_audit.jsonl").open("a") as stream:
        stream.write(',"response":{"tool":"finish","arguments":{}}}\n')
    assert read_history(root, "event:5")["output"]["tool"] == "finish"


def test_images_cannot_escape_episode_and_missing_files_are_reported(tmp_path):
    root = recorded_episode(tmp_path)
    outside = tmp_path / "private.png"
    Image.new("RGB", (24, 16)).save(outside)
    (root / "link.png").symlink_to(outside)
    append(root / "provider_inputs.jsonl", dict(session_id="extra", request_id="extra", role="point",
        attached_image_refs=["img_outside", "img_link", "img_missing"],
        image_paths={"img_outside": str(outside), "img_link": str(root / "link.png"),
                     "img_missing": str(root / "missing.png")}))
    view = read_history(root)
    assert view["input_images"] == []
    assert "3 recorded image(s) unavailable" in view["summary"]


def test_redaction_changes_display_only(tmp_path, monkeypatch):
    root = recorded_episode(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "private-example-key")
    append(root / "provider_audit.jsonl", dict(request_id="p2",
        response={"tool": "finish", "arguments": {"text": "private-example-key"}}))
    before = (root / "provider_audit.jsonl").read_bytes()
    view = read_history(root)
    assert view["output"]["arguments"]["text"] == "[redacted]"
    assert (root / "provider_audit.jsonl").read_bytes() == before


def test_bearer_redaction_preserves_nested_display_data_and_raw_records(tmp_path):
    root = recorded_episode(tmp_path)
    arguments = {"text": "Authorization: Bearer fake-display-token",
                 "details": [{"error": "authorization=Bearer fake-nested-token"}, None, False, 3]}
    append(root / "provider_audit.jsonl", dict(request_id="p2",
        response={"tool": "finish", "arguments": arguments}))
    before = {path: path.read_bytes() for path in root.glob("*.jsonl")}
    view = read_history(root)
    assert view["output"] == {"tool": "finish", "arguments": {
        "text": "Authorization: Bearer [redacted]",
        "details": [{"error": "authorization=Bearer [redacted]"}, None, False, 3]}}
    assert json.loads(json.dumps(view)) == view
    assert all(path.read_bytes() == content for path, content in before.items())


def test_empty_history_and_events_without_provider_records(tmp_path):
    assert read_history(None)["count"] == 0
    assert read_history(tmp_path)["count"] == 0
    append(tmp_path / "events.jsonl", dict(seq=1, kind="agent_input", role="point", session_id="p", instruction="Inspect"))
    view = read_history(tmp_path)
    assert view["input"]["instruction"] == "Inspect"
    assert view["full_input"]["note"]
