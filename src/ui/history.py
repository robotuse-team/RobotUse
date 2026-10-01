"""Read recorded agent turns and their images without changing execution artifacts."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from functools import lru_cache
import json
from pathlib import Path

from src.utils.logging_utils import redact_data

FILES = ("events.jsonl", "provider_inputs.jsonl", "provider_audit.jsonl")


def _rows(path):
    try:
        with path.open() as stream:
            for line in stream:
                try:
                    value = json.loads(line)
                except ValueError:
                    continue  # A running episode may still be writing its last row.
                if isinstance(value, dict):
                    yield value
    except OSError:
        return


@lru_cache(maxsize=8)
def _load(episode, signatures):
    groups, current, sessions = [], {}, defaultdict(list)
    for event in _rows(episode / FILES[0]):
        sid = event.get("session_id")
        if event.get("kind") == "agent_input":
            group = dict(id=f"event:{event['seq']}", event=event, request={}, results=[])
            groups.append(group)
            sessions[sid].append(group)
            current[sid] = group
        elif sid in current and event.get("kind") == "tool_result":
            current[sid]["results"].append(dict(tool=event.get("tool"), result=event.get("result")))
    counters, images = defaultdict(int), {}
    for request in _rows(episode / FILES[1]):
        sid = request.get("session_id")
        index = counters[sid]
        counters[sid] += 1
        if index < len(sessions[sid]):
            sessions[sid][index]["request"] = request
        else:
            groups.append(dict(id="request:" + str(request.get("request_id", len(groups))),
                               event={}, request=request, results=[]))
        images.update(request.get("image_paths", {}))
    responses = {row["request_id"]: row for row in _rows(episode / FILES[2]) if row.get("request_id")}
    for group in groups:
        group["audit"] = responses.get(group["request"].get("request_id"), {})
    return groups, images


def _references(value):
    if isinstance(value, str) and value.startswith("img_"):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _references(item)
    elif isinstance(value, list):
        for item in value:
            yield from _references(item)


def _gallery(episode, references, paths):
    images, missing = [], []
    for ref in dict.fromkeys(references):
        value = paths.get(ref)
        path = Path(value) if isinstance(value, str) else None
        if path is not None:
            path = (path if path.is_absolute() else episode / path).resolve()
        if (path is None or not path.is_relative_to(episode.resolve()) or not path.is_file()
                or path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp")):
            missing.append(ref)
            continue
        name = str(path.relative_to(episode.resolve()))
        if path.name.endswith("_agentview.png"):
            name = "Front camera"
        elif path.name.endswith("_robot0_eye_in_hand.png"):
            name = "Wrist camera"
        images.append((str(path), name))
    return images, missing


def _content(value):
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return value


def read_history(episode: Path | None, turn_id=None, *, follow_latest=False, offset=0):
    empty = dict(choices=[], selected=None, summary="No recorded turns yet.", input={}, output={},
                 results=[], input_images=[], result_images=[], full_input={}, index=0, count=0)
    if episode is None:
        return empty
    signatures = []
    for filename in FILES:
        try:
            stat = (episode / filename).stat()
            signatures.append((stat.st_mtime_ns, stat.st_size))
        except OSError:
            signatures.append(None)
    groups, paths = _load(episode, tuple(signatures))
    if not groups:
        return empty
    index = next((i for i, group in enumerate(groups) if group["id"] == turn_id), len(groups) - 1)
    if follow_latest:
        index = len(groups) - 1
    index = max(0, min(len(groups) - 1, index + offset))
    group = groups[index]
    event, request, audit = group["event"], group["request"], group["audit"]
    role = str(request.get("role", event.get("role", "agent"))).capitalize()
    status = "Response recorded" if audit else "No response recorded"
    summary = f"Turn {index + 1} / {len(groups)} · {role} · {status}"
    if request.get("time_unix_s") is not None:
        timestamp = datetime.fromtimestamp(request["time_unix_s"], timezone.utc)
        summary += " · " + timestamp.strftime("%H:%M:%S UTC")
    input_refs = request.get("attached_image_refs", event.get("image_refs", []))
    input_images, missing_inputs = _gallery(episode, input_refs, paths)
    result_images, missing_results = _gallery(episode, _references(group["results"]), paths)
    missing = len(set(missing_inputs + missing_results))
    if missing:
        summary += f" · {missing} recorded image(s) unavailable"
    full_input = {key: request[key] for key in ("request_id", "provider", "model", "tools") if key in request}
    full_input["messages"] = [dict(role=m.get("role"), content=[_content(text) for text in m.get("text", [])])
                              for m in request.get("messages", [])]
    if not request:
        full_input = {"note": "LLM input was not recorded; the input above comes from agent events."}
    result = dict(choices=[(f"{i + 1} · {g['request'].get('role', g['event'].get('role', 'agent')).capitalize()}", g["id"])
                           for i, g in enumerate(groups)],
        selected=group["id"], summary=summary,
        input={key: event[key] for key in ("instruction", "observation", "execution_budget", "step") if key in event},
        output=audit.get("response", {}), results=group["results"], input_images=input_images,
        result_images=result_images, full_input=full_input, index=index, count=len(groups))
    # Redact presentation only. Never rewrite provider records or images.
    return redact_data(result)
