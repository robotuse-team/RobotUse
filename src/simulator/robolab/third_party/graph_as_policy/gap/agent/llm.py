"""Generic LLM provider layer for the codegen pipeline.

One config (:class:`LlmConfig`) + two calls:

- :func:`complete` — system + messages → assistant text.
- :func:`complete_with_tools` — same, with a native tool-use loop. The
  caller supplies canonical tool descriptors
  (``{"name", "description", "input_schema"}``) and a ``tool_handler``;
  the loop runs the handler for each tool call and feeds results back
  until the model stops calling tools (or ``max_rounds`` is hit).

Providers:

- ``openrouter`` (default) — OpenRouter's OpenAI-compatible
  chat-completions API over httpx (429 backoff, retry-on-5xx,
  ``reasoning_content`` coalescing); tools via the OpenAI tools API.
  Base URL defaults to ``https://openrouter.ai/api/v1`` and the key is
  read from ``OPENROUTER_API_KEY`` (or ``api_key``); ``endpoint`` may
  point at any other OpenAI-compatible server.
- ``vertex`` — Vertex AI via ``google-genai`` (Gemini models), with a
  native function-calling tool loop. Lazy import; install the
  ``[vertex]`` extra.

A small disk cache (keyed on provider+model+prompt hash) memoizes
:func:`complete` responses when ``cache_dir`` (or ``GAP_LLM_CACHE_DIR``)
is set. ``GAP_LLM_NO_CACHE=1`` bypasses it.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import os
import random
import weakref
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_INITIAL_BACKOFF = 2.0  # seconds

#: Provider default models. Both providers default to
#: ``gemini-3.1-flash-lite-preview`` — the verified default for ``gap
#: generate``: ~3× faster codegen than 2.5-flash on this repo's 4-subgraph
#: quickstart, picks the hand-curated topology, and a fresh clean-env run
#: on ``libero_object_all_variance/0`` ended SUCCESS with ``target_held``
#: PASSED. On ``openrouter`` the slug may need a ``google/`` prefix
#: depending on the account; override via ``GAP_LLM_MODEL`` / ``--model``
#: / ``llm.model``.
DEFAULT_MODELS: dict[str, str] = {
    "openrouter": "gemini-3.1-flash-lite-preview",
    "vertex": "gemini-3.1-flash-lite-preview",
}

# Per-event-loop semaphores for rate limiting concurrent LLM requests.
# Keyed by loop (weakly) because asyncio primitives bind to the loop they
# first block on — the coder meta-tool runs nested completions on a
# worker-thread loop and must not share the caller's semaphore object.
_llm_semaphores: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, dict[int, asyncio.Semaphore]] = (
    weakref.WeakKeyDictionary()
)

# Cached Vertex AI Gemini clients (keyed by (project_id, region)).
_vertex_gemini_clients: dict[tuple[str, str], Any] = {}

# Test seam: when set, httpx clients are built with this transport
# (httpx.MockTransport in the unit tests).
_HTTPX_TRANSPORT: Any | None = None


def default_provider() -> str:
    """``$GAP_LLM_PROVIDER`` when set, else ``"openrouter"``.

    Lets a shell pin its provider once (e.g. ``export
    GAP_LLM_PROVIDER=vertex GAP_LLM_MODEL=gemini-...``) so the bare
    ``gap generate "<task>"`` works without per-call flags. Precedence
    stays: ``--provider`` flag > config YAML > this env default.
    """
    return os.environ.get("GAP_LLM_PROVIDER", "").strip().lower() or "openrouter"


def default_model() -> str | None:
    """``$GAP_LLM_MODEL`` when set, else ``None`` (provider default)."""
    return os.environ.get("GAP_LLM_MODEL", "").strip() or None


@dataclass
class LlmConfig:
    """LLM API configuration for the codegen pipeline."""

    provider: str = field(default_factory=default_provider)
    """``"openrouter"`` (default) | ``"vertex"``. The dataclass default
    honors ``$GAP_LLM_PROVIDER``."""

    model: str | None = field(default_factory=default_model)
    """Model id. ``None`` uses the provider default (both providers
    default to ``gemini-3.1-flash-lite-preview``). The dataclass default
    honors ``$GAP_LLM_MODEL``."""

    endpoint: str | None = None
    """OpenAI-compatible base URL (``http://host:port/v1``) or a full
    ``.../chat/completions`` URL — both are accepted. ``None`` means
    OpenRouter (``https://openrouter.ai/api/v1``)."""

    api_key: str | None = None
    """API key. ``None`` falls back to ``OPENROUTER_API_KEY`` for the
    ``openrouter`` provider; ``vertex`` authenticates via Google ADC."""

    project_id: str | None = None
    """GCP project for Vertex AI."""
    region: str | None = None
    """Vertex AI region (e.g. ``"global"``, ``"us-central1"``)."""

    temperature: float | None = 0.7
    """Sampling temperature. Not forwarded to models that reject
    sampling parameters; ``None`` always omits it."""

    max_tokens: int = 20480
    max_concurrent_requests: int = 4

    cache_dir: str | Path | None = None
    """Disk response cache directory. ``None`` falls back to the
    ``GAP_LLM_CACHE_DIR`` env var; unset means caching is disabled."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_model(config: LlmConfig) -> str:
    if config.model:
        return config.model
    default = DEFAULT_MODELS.get(config.provider)
    if default is None:
        raise ValueError(
            f"LlmConfig.model must be set for provider={config.provider!r} "
            f"(no provider default)"
        )
    return default


def _is_claude_model(model: str) -> bool:
    return "claude" in model.lower()


def _reject_claude_on_vertex(model: str) -> None:
    """Vertex now serves Gemini only (the anthropic dep was removed)."""
    if _is_claude_model(model):
        raise ValueError(
            f"vertex serves Gemini models only (got {model!r}); "
            "Claude-on-Vertex was removed with the anthropic dependency. "
            "Use a gemini-* model, or route Claude via the openrouter provider."
        )


def _get_semaphore(max_concurrent: int) -> asyncio.Semaphore:
    """Return this event loop's semaphore for *max_concurrent*."""
    loop = asyncio.get_running_loop()
    by_limit = _llm_semaphores.get(loop)
    if by_limit is None:
        by_limit = {}
        _llm_semaphores[loop] = by_limit
    sem = by_limit.get(max_concurrent)
    if sem is None:
        sem = asyncio.Semaphore(max_concurrent)
        by_limit[max_concurrent] = sem
        logger.debug("LLM concurrency semaphore set to %d", max_concurrent)
    return sem


def _httpx_client() -> httpx.AsyncClient:
    kwargs: dict[str, Any] = {"timeout": 600.0}
    if _HTTPX_TRANSPORT is not None:
        kwargs["transport"] = _HTTPX_TRANSPORT
    return httpx.AsyncClient(**kwargs)


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _stringify_tool_result(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value)
    except Exception:
        return str(value)


# ---------------------------------------------------------------------------
# Disk response cache
# ---------------------------------------------------------------------------


def _cache_dir(config: LlmConfig) -> Path | None:
    if config.cache_dir is not None:
        return Path(config.cache_dir)
    env = os.environ.get("GAP_LLM_CACHE_DIR")
    return Path(env) if env else None


def _cache_disabled() -> bool:
    """When ``GAP_LLM_NO_CACHE=1`` (or truthy), bypass the prompt cache.

    The cache only correctly memoizes deterministic calls — at temp>0 it
    collapses K stochastic samples to one response.
    """
    v = os.environ.get("GAP_LLM_NO_CACHE", "")
    return v.lower() in ("1", "true", "yes", "on")


def _cache_key(config: LlmConfig, system: str, messages: list[dict]) -> str:
    """Deterministic hash over provider + model + generation knobs + prompt."""
    h = hashlib.sha256()
    h.update(config.provider.encode())
    h.update(_resolve_model(config).encode())
    h.update(repr((config.temperature, config.max_tokens)).encode())
    h.update(system.encode())
    for m in messages:
        h.update(str(m.get("role", "")).encode())
        h.update(str(m.get("content", "")).encode())
    return h.hexdigest()


def _cache_load(config: LlmConfig, key: str) -> str | None:
    if _cache_disabled():
        return None
    root = _cache_dir(config)
    if root is None:
        return None
    path = root / f"{key}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())["raw"]
    except Exception:
        return None


def _cache_save(config: LlmConfig, key: str, raw: str) -> None:
    if _cache_disabled():
        return
    root = _cache_dir(config)
    if root is None:
        return
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{key}.json").write_text(json.dumps({"raw": raw}))


# ---------------------------------------------------------------------------
# OpenRouter path (OpenAI-compatible chat-completions over httpx)
# ---------------------------------------------------------------------------


def _openai_url(config: LlmConfig) -> str:
    """Accept both a base URL (``http://host:port/v1``) and a full
    chat/completions URL. ``endpoint`` defaults to OpenRouter."""
    endpoint = config.endpoint or "https://openrouter.ai/api/v1"
    if "/chat/completions" in endpoint:
        return endpoint
    return endpoint.rstrip("/") + "/chat/completions"


def _openai_headers(config: LlmConfig) -> dict[str, str]:
    headers: dict[str, str] = {}
    api_key = config.api_key or os.environ.get("OPENROUTER_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


async def _post_openai_chat(config: LlmConfig, request_body: dict) -> dict:
    """POST with the source's retry/backoff structure: 429 → exponential
    backoff with jitter; transport errors and 5xx → retry; other 4xx →
    raise immediately."""
    headers = _openai_headers(config)
    data: dict = {}
    async with _httpx_client() as client:
        for attempt in range(_MAX_RETRIES + 1):
            try:
                response = await client.post(
                    _openai_url(config), json=request_body, headers=headers,
                )
                response.raise_for_status()
                data = response.json()
                break
            except (httpx.ConnectError, httpx.PoolTimeout, httpx.ReadTimeout,
                    httpx.WriteTimeout, httpx.HTTPStatusError) as e:
                retryable = True
                if isinstance(e, httpx.HTTPStatusError):
                    status = e.response.status_code
                    if status == 429:
                        backoff = _INITIAL_BACKOFF * (2 ** (attempt + 1)) + random.uniform(0, 2)
                        logger.warning(
                            "LLM rate-limited (429, attempt %d/%d), retrying in %.1fs",
                            attempt + 1, _MAX_RETRIES + 1, backoff,
                        )
                        await asyncio.sleep(backoff)
                        continue
                    elif status < 500:
                        retryable = False
                if not retryable:
                    raise
                if attempt == _MAX_RETRIES:
                    raise
                backoff = _INITIAL_BACKOFF * (2 ** attempt) + random.uniform(0, 1)
                logger.warning(
                    "LLM request failed (attempt %d/%d, %s: %s), retrying in %.1fs",
                    attempt + 1, _MAX_RETRIES + 1, type(e).__name__, e, backoff,
                )
                await asyncio.sleep(backoff)
    return data


def _openai_request_body(
    config: LlmConfig,
    system: str,
    messages: list[dict],
    tools: list[dict] | None = None,
) -> dict:
    api_messages: list[dict] = [{"role": "system", "content": system}]
    api_messages.extend(messages)
    body: dict[str, Any] = {
        "model": _resolve_model(config),
        "messages": api_messages,
        "max_tokens": config.max_tokens,
    }
    if config.temperature is not None:
        body["temperature"] = config.temperature
    if tools:
        body["tools"] = [_to_openai_tool(t) for t in tools]
    return body


def _to_openai_tool(tool: dict) -> dict:
    """Canonical tool descriptor → OpenAI tools-API shape."""
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
        },
    }


def _openai_message_content(msg: dict) -> str:
    content = msg.get("content")
    # vLLM with a reasoning parser may set content=None when the entire
    # response was thinking tokens — coalesce so downstream parsers don't
    # crash on .splitlines() etc.
    if content is None:
        content = msg.get("reasoning_content") or msg.get("reasoning") or ""
    if content is None:
        content = ""
    return content


async def _call_openrouter_async(
    config: LlmConfig, system: str, messages: list[dict],
) -> str:
    data = await _post_openai_chat(config, _openai_request_body(config, system, messages))
    choices = data.get("choices", [])
    if not choices:
        return ""
    return _openai_message_content(choices[0].get("message", {}) or {})


async def _openrouter_tool_loop(
    config: LlmConfig,
    system: str,
    messages: list[dict],
    tools: list[dict],
    tool_handler: Callable[[str, dict], Any],
    max_rounds: int,
) -> str:
    api_messages = [dict(m) for m in messages]
    for _round in range(max_rounds):
        body = _openai_request_body(config, system, api_messages, tools=tools)
        data = await _post_openai_chat(config, body)
        choices = data.get("choices", [])
        if not choices:
            return ""
        msg = choices[0].get("message", {}) or {}
        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            return _openai_message_content(msg)

        api_messages.append({
            "role": "assistant",
            "content": msg.get("content"),
            "tool_calls": tool_calls,
        })
        for tc in tool_calls:
            fn = tc.get("function", {}) or {}
            name = fn.get("name", "")
            try:
                kwargs = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                kwargs = {}
            try:
                out = await _maybe_await(tool_handler(name, kwargs))
                content = _stringify_tool_result(out)
            except Exception as e:
                content = f"{type(e).__name__}: {e}"
            api_messages.append({
                "role": "tool",
                "tool_call_id": tc.get("id", ""),
                "content": content,
            })

    raise RuntimeError(
        f"tool-use loop exceeded {max_rounds} rounds without a final response"
    )


# ---------------------------------------------------------------------------
# Vertex AI path
# ---------------------------------------------------------------------------

_VERTEX_HINT = (
    "Vertex AI support requires the [vertex] extra: "
    "pip install 'graph-as-policy[vertex]'"
)


def _vertex_project(config: LlmConfig) -> str:
    """Resolve the GCP project: config wins, then the documented env var.

    ``llm.project_id`` from a config YAML takes precedence; otherwise
    ``$GOOGLE_CLOUD_PROJECT`` (the README's documented knob — also what
    google-genai reads natively), so ``--provider vertex`` works with no
    config file. Resolved here, *before* the client cache key, so two
    shells with different env projects never share a cached client.
    """
    return (
        config.project_id
        or os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip()
    )


def _vertex_region(config: LlmConfig) -> str:
    """Resolve the Vertex region: config > documented env vars > global."""
    return (
        config.region
        or os.environ.get("GOOGLE_CLOUD_REGION", "").strip()
        or os.environ.get("GOOGLE_CLOUD_LOCATION", "").strip()
        or "global"
    )


def _vertex_gemini_client(config: LlmConfig) -> Any:
    try:
        from google import genai
    except ImportError as e:  # pragma: no cover - import-guarded
        raise ImportError(_VERTEX_HINT) from e
    project = _vertex_project(config)
    region = _vertex_region(config)
    key = (project, region)
    client = _vertex_gemini_clients.get(key)
    if client is None:
        client = genai.Client(vertexai=True, project=project, location=region)
        _vertex_gemini_clients[key] = client
    return client


async def _call_vertex_async(
    config: LlmConfig, system: str, messages: list[dict],
) -> str:
    """Call Vertex AI via ``google.genai.Client`` (Gemini, sync via
    ``to_thread``)."""
    model = _resolve_model(config)
    _reject_claude_on_vertex(model)

    try:
        from google.genai import types
    except ImportError as e:  # pragma: no cover - import-guarded
        raise ImportError(_VERTEX_HINT) from e

    client = _vertex_gemini_client(config)
    contents = [
        types.Content(
            role="model" if m.get("role") == "assistant" else "user",
            parts=[types.Part.from_text(text=str(m.get("content", "")))],
        )
        for m in messages
    ]
    gen_config_kwargs: dict[str, Any] = {
        "max_output_tokens": config.max_tokens,
        "system_instruction": system,
    }
    if config.temperature is not None:
        gen_config_kwargs["temperature"] = config.temperature
    response = await asyncio.to_thread(
        client.models.generate_content,
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(**gen_config_kwargs),
    )
    return response.text or ""


async def _vertex_gemini_tool_loop(
    config: LlmConfig,
    system: str,
    messages: list[dict],
    tools: list[dict],
    tool_handler: Callable[[str, dict], Any],
    max_rounds: int,
) -> str:
    """Native Gemini (google-genai) function-calling loop on Vertex.

    Mirrors the openrouter loop: tools are declared up front, automatic
    function calling is disabled so we drive each round trip and dispatch
    through *tool_handler* ourselves. Canonical tool descriptors translate
    to Gemini ``FunctionDeclaration``s by mapping ``input_schema`` →
    ``parameters``.
    """
    try:
        from google.genai import types
    except ImportError as e:  # pragma: no cover - import-guarded
        raise ImportError(_VERTEX_HINT) from e

    client = _vertex_gemini_client(config)
    model = _resolve_model(config)
    gemini_tools = [types.Tool(function_declarations=[
        {
            "name": t["name"],
            "description": t.get("description", ""),
            "parameters": t.get(
                "input_schema", {"type": "object", "properties": {}},
            ),
        }
        for t in tools
    ])]
    base_kwargs: dict[str, Any] = {
        "max_output_tokens": config.max_tokens,
        "system_instruction": system,
        "automatic_function_calling": types.AutomaticFunctionCallingConfig(
            disable=True,
        ),
    }
    if config.temperature is not None:
        base_kwargs["temperature"] = config.temperature

    def _gen_config(force_text: bool):
        kwargs = dict(base_kwargs, tools=gemini_tools)
        if force_text:
            # Final turn: forbid further tool calls so the model must emit
            # the answer as text instead of looping on tools forever.
            kwargs["tool_config"] = types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode=types.FunctionCallingConfigMode.NONE,
                ),
            )
        return types.GenerateContentConfig(**kwargs)

    contents = [
        types.Content(
            role="model" if m.get("role") == "assistant" else "user",
            parts=[types.Part.from_text(text=str(m.get("content", "")))],
        )
        for m in messages
    ]

    async def _generate(force_text: bool):
        return await asyncio.to_thread(
            client.models.generate_content,
            model=model,
            contents=contents,
            config=_gen_config(force_text),
        )

    # One extra, tool-free turn beyond max_rounds to coax a final answer.
    for round_idx in range(max_rounds + 1):
        force_text = round_idx == max_rounds
        response = await _generate(force_text)
        candidate = (response.candidates or [None])[0]
        content = getattr(candidate, "content", None)
        parts = list(getattr(content, "parts", None) or [])
        calls = [p.function_call for p in parts
                 if getattr(p, "function_call", None)]
        if not calls or force_text:
            if calls and force_text:
                logger.warning(
                    "vertex gemini tool-use loop hit %d rounds; forced a "
                    "final tool-free completion", max_rounds,
                )
            return response.text or ""

        # Echo the model turn verbatim, then a user turn of function results.
        if content is not None:
            contents.append(content)
        result_parts = []
        for call in calls:
            name = getattr(call, "name", "") or ""
            args = dict(getattr(call, "args", {}) or {})
            try:
                out = await _maybe_await(tool_handler(name, args))
                payload = {"result": _stringify_tool_result(out)}
            except Exception as e:
                payload = {"error": f"{type(e).__name__}: {e}"}
            result_parts.append(
                types.Part.from_function_response(name=name, response=payload)
            )
        contents.append(types.Content(role="user", parts=result_parts))

    # Unreachable: the force_text iteration above always returns.
    raise RuntimeError(
        f"tool-use loop exceeded {max_rounds} rounds without a final response"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def complete(
    config: LlmConfig,
    *,
    system: str,
    messages: list[dict],
) -> str:
    """One LLM completion: system + messages → assistant text.

    Checks the disk cache first (when configured); rate-limits via the
    module semaphore; retries (vertex transport errors) with exponential
    backoff.
    """
    key = _cache_key(config, system, messages)
    cached = _cache_load(config, key)
    if cached is not None:
        logger.info("LLM cache hit (%s…)", key[:12])
        return cached

    sem = _get_semaphore(config.max_concurrent_requests)

    if config.provider == "vertex":
        call = _call_vertex_async
    elif config.provider == "openrouter":
        call = _call_openrouter_async
    else:
        raise ValueError(f"unknown LLM provider {config.provider!r}")

    # The openrouter path carries its own per-status retry logic; wrap the
    # vertex SDK path in the generic retry loop the source applied to it.
    retries = [0] if config.provider == "openrouter" else range(_MAX_RETRIES + 1)
    content = ""
    async with sem:
        for attempt in retries:
            try:
                content = await call(config, system, messages)
                break
            except Exception as e:
                if attempt == _MAX_RETRIES or config.provider == "openrouter":
                    raise
                backoff = _INITIAL_BACKOFF * (2 ** attempt) + random.uniform(0, 1)
                logger.warning(
                    "LLM request failed (attempt %d/%d, %s: %s), retrying in %.1fs",
                    attempt + 1, _MAX_RETRIES + 1, type(e).__name__, e, backoff,
                )
                await asyncio.sleep(backoff)

    _cache_save(config, key, content)
    logger.info("LLM cache miss — saved (%s…)", key[:12])
    return content


async def complete_with_tools(
    config: LlmConfig,
    *,
    system: str,
    messages: list[dict],
    tools: list[dict],
    tool_handler: Callable[[str, dict], Any],
    max_rounds: int = 6,
) -> str:
    """LLM completion with a native tool-use loop on every provider.

    Args:
        config: Provider config.
        system: System prompt text.
        messages: Conversation history (``{"role", "content"}`` dicts).
        tools: Canonical tool descriptors
            (``{"name", "description", "input_schema"}``). Translated to
            the OpenAI tools shape (openrouter) or Gemini function
            declarations (vertex).
        tool_handler: ``handler(name, input_dict) -> Any`` — sync or
            async. Exceptions become error tool-results fed back to the
            model.
        max_rounds: Cap on model⇄tool round trips.

    Tool-loop responses are never disk-cached (tool side effects make
    memoization unsound).
    """
    if not tools:
        return await complete(config, system=system, messages=messages)

    sem = _get_semaphore(config.max_concurrent_requests)
    async with sem:
        if config.provider == "openrouter":
            return await _openrouter_tool_loop(
                config, system, messages, tools, tool_handler, max_rounds,
            )
        if config.provider == "vertex":
            _reject_claude_on_vertex(_resolve_model(config))
            return await _vertex_gemini_tool_loop(
                config, system, messages, tools, tool_handler, max_rounds,
            )
    raise ValueError(f"unknown LLM provider {config.provider!r}")


__all__ = ["DEFAULT_MODELS", "LlmConfig", "complete", "complete_with_tools"]
