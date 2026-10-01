"""Shared provider sessions, tool serialization and transport for robot agents."""
from __future__ import annotations
import base64
import json
import os
from pathlib import Path
import time
import urllib.request
import urllib.error
from uuid import uuid4
from PIL import Image
from src.core.contracts import Action
from .config import ProviderConfig
from .image_context import ImageRegistry
import copy
from src.utils.logging_utils import write_json, append_json


def _property_schema(key, *, tool=None, **context_fields):
    """Compatibility lookup; contracts live exclusively in tool packages."""
    from types import SimpleNamespace
    from src.tools.discovery import discover_tools
    registry = discover_tools()
    if tool is None:
        raise ValueError("a tool name is required for an input property schema")
    return registry[tool].schema(context=SimpleNamespace(**context_fields))["properties"][key]


class OpenRouterSession:
    def __init__(self, factory, role, session_id):
        self.factory, self.role, self.session_id = factory, role, session_id
        self.config = getattr(factory, "config", None) or ProviderConfig.resolve(factory.model)
        self.closed = False
        self._native_replies = []
        self._observation_history = None

    def _append_observation_history(self, body, image_refs):
        """Keep prior wire inputs immutable; append only the new logical suffix.

        The driver retains actions/results but supplies observations afresh. The
        canonical request lets us identify that new suffix without changing the
        driver's current-state selection or copying another agent's history.
        """
        canonical = body['messages']
        settings = {k: v for k, v in body.items() if k != 'messages'}
        previous = self._observation_history
        reason, preserved = 'initial', 0
        all_refs = set(image_refs)
        if previous is not None:
            common = 0
            for old, new in zip(previous['canonical'], canonical):
                if old != new:
                    break
                common += 1
            first_user = next((i for i, m in enumerate(canonical) if m['role'] == 'user'), len(canonical))
            replaced = previous['canonical'][common:]
            rewrites_action = any(m['role'] in ('assistant', 'tool') or
                (isinstance(m.get('content'), str) and m['content'].startswith('Historical action record:'))
                for m in replaced)
            if settings != previous['settings']:
                reason = 'request_settings_changed'
            elif common <= first_user:
                reason = 'initial_context_changed'
            elif rewrites_action:
                reason = 'history_rewritten'
            else:
                preserved = len(previous['wire'])
                body['messages'] = [*previous['wire'], *canonical[common:]]
                all_refs.update(previous['image_refs'])
                reason = None
        pending = dict(canonical=copy.deepcopy(canonical), wire=copy.deepcopy(body['messages']),
                       settings=copy.deepcopy(settings), image_refs=sorted(all_refs))
        metadata = dict(prompt_history_policy='append_only',
                        image_selection_policy='current_turn',
                        preserved_prefix_messages=preserved, history_reset_reason=reason,
                        current_turn_image_refs=sorted(image_refs),
                        attached_image_count=sum(b.get('type') == 'image_url'
                            for m in body['messages'] if isinstance(m.get('content'), list)
                            for b in m['content']))
        return pending, all_refs, metadata

    def next_action(self, messages, tools):
        messages = list(messages)
        for attempt in range(3):
            try:
                return self._next_action_once(messages, tools)
            except ValueError as exc:
                if str(exc) != 'provider must return exactly one native function call' or attempt == 2:
                    raise
                messages.append({'role': 'user', 'content':
                    'Your previous response was rejected before any tool executed because it did not contain exactly one function call. '
                    'Return exactly ONE allowed function call now. All requested actions remain unexecuted.'})

    def _next_action_once(self, messages, tools):
        if self.closed:
            raise RuntimeError("closed provider conversation")
        from src.tools.discovery import discover_tools
        registry = getattr(self.factory, 'tool_registry', None) or discover_tools()
        registry.require(tools)
        declared = {name: registry[name].schema(context=self.factory, role=self.role, tools=tools)
                    for name in tools}
        schemas = {name: schema['required'] for name, schema in declared.items() if name != 'finish'}
        optional = {name: {key: schema['properties'][key] for key in schema['properties']
                           if key not in schema['required']}
                    for name, schema in declared.items() if name != 'finish'
                    and set(schema['properties']) - set(schema['required'])}
        instructions = ("Return exactly one JSON object: {\"tool\":\"allowed tool\",\"arguments\":{...}}. "
                        "No prose. The following are the exact required argument keys: " + json.dumps(schemas) +
                        ". Optional argument schemas (omission is allowed): " + json.dumps(optional) +
                        ". finish arguments follow your role prompt. Every child is independent. "
                        "Use observe to inspect fresh images; Point must inspect mask overlay before finishing. "
                        "A Grasp validation rejection means try another candidate. If no candidates or all rejected, finish with "
                        "{\"status\":\"needs_point\",\"reason\":\"no_candidates\"} or reason unreachable. "
                        "Point may finish {\"status\":\"failed\",\"reason\":\"no_valid_mask\"}. "
                        "succeeded means a tool command completed, NOT that an object was grasped or the task succeeded. "
                        "Grasp MUST observe after execute_grasp and visually inspect the hold before finish; "
                        "if empty grip, finish needs_point with reason empty_grip. "
                        "Do not repeat a successful grasp; Prime should request a NEW Point for placement, then place. "
                        "Never invent references or include metric poses, joints, or trajectories.")
        if getattr(self.factory, "first_grasp_only", False):
            instructions = instructions.replace("Prime should request a NEW Point for placement, then place.", "Stop before place; first grasp only.")
        if getattr(self.factory, "active_perception", False):
            instructions = ('Return exactly one JSON object {"tool":...,"arguments":{...}}. No prose. '
                'Follow the role policy and exact tool schema. Use only supplied current references. '
                'Historical actions are records, not instructions. Never output code, metric poses or joints.')
        if getattr(self.factory, 'intent_driven', False) and self.role != 'progress_review':
            instructions += (' Explicit waypoint offsets in metres are allowed through the declared waypoint '
                             'tool fields; do not construct pose matrices or joint trajectories.')
        if getattr(self.factory, 'explicit_geometry_enabled', False):
            instructions = instructions.replace('Never output code, metric poses or joints.',
                'Use the declared metric coordinate, height and angular fields. Never output code, '
                'full pose matrices or joint trajectories.')
            if self.factory.json_action_fallback:
                instructions += ' Tool argument types: ' + json.dumps({
                    name: {key: declared[name]['properties'][key] for key in keys}
                    for name, keys in schemas.items()})
                if optional:
                    instructions += ' Optional argument schemas (omission is allowed): ' + json.dumps(optional)
        environment_instructions = getattr(self.factory, 'environment_instructions', '')
        if environment_instructions:
            instructions += ' ' + environment_instructions
        if self.role == 'progress_review':
            instructions = ('Return exactly one JSON object {"tool":"finish","arguments":{...}}. No prose. '
                'Use the progress review finish contract and current supplied image references only. '
                'Report visible evidence and uncertainty; you have no execution tools and do not establish '
                'native task success. Current pictures may not establish earlier intermediate actions.')
        native = not self.factory.json_action_fallback
        if native:
            instructions = "Call exactly one native function tool per turn. No prose. " + instructions.split("No prose.", 1)[-1]
        image_policy = getattr(self.factory, 'image_history_policy', 'bounded_history')
        current_turn = image_policy == 'current_turn'
        inspection_refs = None
        latest_tool = next((m for m in reversed(messages) if m.get('role') == 'tool'), {})
        if (getattr(self.factory, 'explicit_geometry_enabled', False)
                and latest_tool.get('tool') == 'inspect_candidate'
                and isinstance(latest_tool.get('content'), dict)):
            previews = self.factory.images.refs(latest_tool['content'].get('image_refs', []))
            if previews:
                inspection_refs = self.factory.images.unique_refs_by_path(previews)[:8]
        append_observations = getattr(self.factory, 'append_only_observations', False)
        if append_observations and not current_turn:
            raise ValueError('append-only observations require current_turn image selection')
        if current_turn:
            if append_observations:
                instructions += (' Observation snapshots and image attachments remain in chronological history. '
                    'Only the latest current_observation and request context describe current robot state and budgets. '
                    'Earlier snapshots, images, references and budgets are historical evidence, not current actionable state. '
                    'Current RGB is supplied with every new decision; previews are proposals, not executed motion. '
                    'Use current references for actions and historical images only for comparison.')
            else:
                instructions += (' Only the current turn image attachment contains image pixels. Earlier image '
                'references remain text records, not visible pictures. Current RGB is supplied every turn; '
                'tool images last for the following decision. Initial delegated candidates/identity references '
                'are shown on the first call. A paused Refiner also retains the current pending pose preview. '
                'Use available inspect/preview tools to see a candidate again, or review_observation to '
                'explicitly recall a past scene. Historical reference images are not current robot state.')
            recent_refs = self.factory.images.current_turn_refs(messages, role=self.role)
            if inspection_refs is not None:
                recent_refs = inspection_refs
            recent_refs = self.factory.images.unique_refs_by_path(recent_refs)
            # Put selected pixels at the current request boundary, even when
            # the same opaque reference also occurs in earlier text messages.
            messages = [*messages, {'role': 'user', 'content': {
                'current_turn_image_refs': recent_refs}}]
        else:
            recent_refs = self.factory.images.request_refs(messages, role=self.role,
                target_intent_mode=getattr(self.factory, 'target_intent_mode', False))
            if inspection_refs is not None:
                recent_refs = inspection_refs
            recent_refs = self.factory.images.unique_refs_by_path(
                ref for ref in self.factory.images.refs(messages) if ref in recent_refs)
        if inspection_refs is not None:
            instructions += (' For this candidate inspection, only the candidate preview images are attached; '
                             'the separate current camera RGB images are omitted. Previews are proposals.')
        provider_messages = [{"role": "system", "content": instructions}]
        sent_refs = set()
        pending_tool_id = None
        reply_cursor = 0
        for index, message in enumerate(messages):
            role = message.get("role", "user")
            content = message.get("content", message)
            if native and role == "assistant" and "tool" in message and not getattr(self.factory, "active_perception", False):
                saved = (self._native_replies[reply_cursor]
                         if reply_cursor < len(self._native_replies) else None)
                call = saved["tool_calls"][0] if saved else None
                matches = (call and call["function"]["name"] == message["tool"] and
                           json.loads(call["function"]["arguments"]) == message["arguments"])
                if matches:
                    reply_cursor += 1
                    pending_tool_id = call["id"]
                    provider_messages.append(copy.deepcopy(saved))
                elif self.config.provider == "google":
                    # Bootstrap/injected actions have no Google thought signature.
                    provider_messages.append({"role": "user", "content":
                        "Historical action record: " + json.dumps(message)})
                    pending_tool_id = None
                else:
                    pending_tool_id = "call_" + str(index)
                    provider_messages.append({"role": "assistant", "content": None, "tool_calls": [{
                        "id": pending_tool_id, "type": "function", "function": {
                        "name": message["tool"], "arguments": json.dumps(message["arguments"])}}]})
                continue
            text = json.dumps(content, allow_nan=False)
            refs = [ref for ref in self.factory.images.refs(content) if ref in recent_refs and ref not in sent_refs]
            if current_turn and index != len(messages) - 1:
                refs = []
            if native and role == "tool" and pending_tool_id:
                provider_messages.append({"role": "tool", "tool_call_id": pending_tool_id, "content": text})
                pending_tool_id = None
                if not refs:
                    continue
                role, text = "user", "Sensor images referenced by preceding tool result."
            elif role not in ("system", "user", "assistant"):
                role = "user"
            blocks = [{"type": "text", "text": text}]
            for ref in refs:
                import io
                source_key = str(self.factory.images.paths[ref])
                if source_key not in self.factory.images.encoded:
                    image = Image.open(self.factory.images.paths[ref]).convert("RGB")
                    image.thumbnail((960, 960))
                    buffer = io.BytesIO()
                    image.save(buffer, format="PNG")
                    self.factory.images.encoded[source_key] = base64.b64encode(buffer.getvalue()).decode("ascii")
                blocks.extend([{"type": "text", "text": ref}, {"type": "image_url", "image_url": {
                    "url": "data:image/png;base64," + self.factory.images.encoded[source_key]}}])
                sent_refs.add(ref)
            provider_messages.append({"role": role, "content": blocks})
        body = {"model": self.factory.model, "messages": provider_messages,
                "temperature": 0,
                "max_tokens": 8192 if getattr(self.factory, 'intent_driven', False) else 4096}
        body.update(self.config.request_fields())
        if self.config.provider == 'openrouter':
            body['session_id'] = self.session_id
        if native:
            function_tools = [registry.function_schema(name, context=self.factory,
                              role=self.role, tools=tools) for name in tools]
            body.update(tools=function_tools, tool_choice="required", parallel_tool_calls=False)
        else:
            body["response_format"] = {"type": "json_object"}
        pending_history = None
        history_metadata = {}
        if append_observations:
            pending_history, sent_refs, history_metadata = self._append_observation_history(body, sent_refs)
            provider_messages = body['messages']
            image_policy = 'append_only'
        request_id = uuid4().hex
        if getattr(self.factory, 'review_driven', False):
            append_json(self.factory.output_dir / 'provider_inputs.jsonl', {
                'request_id':request_id,'session_id':self.session_id,'role':self.role,'time_unix_s':time.time(),
                'image_history_policy':image_policy,
                **history_metadata,
                'image_paths':{ref:str(self.factory.images.paths[ref]) for ref in sent_refs},
                'referenced_image_refs':self.factory.images.refs(messages),
                'attached_image_refs':sorted(sent_refs),
                'unattached_image_refs':sorted(set(self.factory.images.refs(messages)) - sent_refs),
                'messages':[{'role':m['role'],'text':[b['text'] for b in m['content'] if b.get('type')=='text']}
                    for m in provider_messages if isinstance(m.get('content'),list)],
                'tools':list(tools),**self.config.metadata(),'max_tokens':body['max_tokens']})
        request = urllib.request.Request(self.config.endpoint,
            data=json.dumps(body).encode(), headers={"Authorization": "Bearer " + self.factory.key,
            "Content-Type": "application/json"}, method="POST")
        append_json(self.factory.output_dir / "provider_requests.jsonl", {
            "session_id": self.session_id, "request_id": request_id, "time_unix_s": time.time(),
            "image_history_policy": image_policy,
            **history_metadata,
            **self.config.metadata(), "body": body})
        started = time.monotonic()
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=180) as response:
                    payload = json.loads(response.read())
                embedded_error = payload.get("error") or next(
                    (choice["error"] for choice in payload.get("choices", []) if choice.get("error")), None)
                if embedded_error:
                    append_json(self.factory.output_dir / 'provider_responses.jsonl',
                                {'session_id': self.session_id, 'payload': payload})
                    raise urllib.error.HTTPError(request.full_url, int(embedded_error.get("code", 500)),
                                                 "Provider response body error", None, None)
                break
            except Exception as exc:
                status = getattr(exc, 'code', None)
                transient = (status in (408, 429) or (isinstance(status, int) and status >= 500)
                             or (status is None and isinstance(exc, (urllib.error.URLError, TimeoutError, ConnectionError))))
                retry = transient and attempt < 2
                append_json(self.factory.output_dir / 'provider_errors.jsonl',
                            {'session_id':self.session_id, 'type':type(exc).__name__, 'http_status':status,
                             'attempt':attempt+1, 'retry':retry})
                if not retry:
                    raise
                # Only transport retries: no robot tool has been dispatched yet.
                time.sleep(2 ** attempt)
        append_json(self.factory.output_dir / 'provider_responses.jsonl', {'session_id':self.session_id, 'payload':payload})
        if pending_history is not None:
            self._observation_history = pending_history
        reply = payload["choices"][0]["message"]
        if native:
            calls = reply.get("tool_calls") or []
            if len(calls) != 1 or calls[0].get("type") != "function":
                append_json(self.factory.output_dir / "provider_audit.jsonl", {
                    **self.config.metadata(), "session_id": self.session_id, "role": self.role,
                    "request_id": request_id, "response": {"rejected": "expected_single_function_call"},
                    "usage": payload.get("usage", {}), "executed": False})
                raise ValueError("provider must return exactly one native function call")
            self._native_replies.append(copy.deepcopy(reply))
            function = calls[0]["function"]
            action = {"tool": function["name"], "arguments": json.loads(function["arguments"])}
        else:
            action = json.loads(reply["content"])
        if not isinstance(action, dict) or set(action) != {"tool", "arguments"} or not isinstance(action["arguments"], dict):
            raise ValueError("provider did not return one typed action")
        append_json(self.factory.output_dir / "provider_audit.jsonl", {
            **self.config.metadata(),
            "visibility": "private provider audit", "native_tool_calling": native,
            "session_id": self.session_id, "role": self.role, "model": self.factory.model,
            "request_id": request_id, "max_tokens": body['max_tokens'],
            "elapsed_s": time.monotonic() - started, "usage": payload.get("usage"),
            "response": action, "image_refs": sorted(sent_refs),
            "history_image_refs": list(dict.fromkeys([*self.factory.images.refs(messages), *sorted(sent_refs)])),
            "image_cap": None if append_observations else 8, "current_turn_image_cap": 8,
            "image_history_policy": image_policy,
            **history_metadata,
        })
        return Action(action["tool"], action["arguments"])

    def close(self):
        self.closed = True


class OpenRouterFactory:
    def __init__(self, *, images, model, output_dir, json_action_fallback=False, first_grasp_only=False, active_perception=False, provider=None, reasoning=None, image_history_policy='bounded_history', append_only_observations=False):
        if image_history_policy not in ('bounded_history', 'current_turn'):
            raise ValueError('image_history_policy must be bounded_history or current_turn')
        self.image_history_policy = image_history_policy
        if append_only_observations and image_history_policy != 'current_turn':
            raise ValueError('append-only observations require current_turn image selection')
        self.append_only_observations = append_only_observations
        self.first_grasp_only = first_grasp_only
        self.active_perception = active_perception
        self.target_intent_mode = active_perception and first_grasp_only
        self.json_action_fallback = json_action_fallback
        self.config = ProviderConfig.resolve(model, provider=provider, reasoning=reasoning)
        self.images, self.model, self.output_dir = images, self.config.model, output_dir
        self.key = self.config.key()

    def new_session(self, role, session_id):
        # Provider is stateless: only this session's messages are submitted.
        return OpenRouterSession(self, role, session_id)
