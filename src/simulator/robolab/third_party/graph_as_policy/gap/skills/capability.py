"""Capability probing — "can this tool bundle run here?" and what follows.

Powers ``gap check``. The atomic question is per tool bundle: are its
deps importable, are its declared :class:`~gap.skills.meta.SkillRequires`
met (GPU, env vars), are its weights cached? Skill runnability *derives*
from that: a skill is blocked exactly by the not-ready bundles that own
its ``allowed_tools`` (``robot.*``/``sim.*`` are connector-provided and
always satisfied once a connector is attached).

Everything here is fast and offline: static SKILL.md parses, one
``nvidia-smi`` subprocess, env-var lookups, per-bundle import probes
(bundles lazy-load their models, so importing ``tools.py`` never loads
weights), and the optional filesystem-only ``weights_cached()`` hook.
No model loads, no downloads, no network.
"""

from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from gap_core.skills.meta import SkillMeta

from .registries import RegistrySet, RegistrySpec
from .validate import load_checkout_extras, validate_checkout

__all__ = [
    "BundleCapability",
    "CheckReport",
    "EnvironmentReport",
    "ProbeResult",
    "SkillRunnability",
    "build_check_report",
    "dep_fix_hint",
    "probe_bundle_import",
    "probe_gpu",
    "probe_llm_providers",
    "probe_vlm_providers",
    "resolve_vlm_env",
]

ProbeStatus = Literal["ok", "missing", "unknown", "error"]

#: Tool-name prefixes registered by the live connector at runtime — never
#: importable statically, always satisfied once a connector is attached.
CONNECTOR_PREFIXES = ("robot", "sim")


@dataclass
class ProbeResult:
    """One probe's verdict."""

    status: ProbeStatus
    detail: str = ""
    fix_hint: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def to_json_dict(self) -> dict[str, str]:
        d = {"status": self.status}
        if self.detail:
            d["detail"] = self.detail
        if self.fix_hint:
            d["fix_hint"] = self.fix_hint
        return d


# ---------------------------------------------------------------------------
# Environment probes
# ---------------------------------------------------------------------------


def probe_gpu(*, timeout: float = 5.0) -> ProbeResult:
    """NVIDIA GPU presence via ``nvidia-smi`` — deliberately torch-free."""
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=timeout,
        )
    except FileNotFoundError:
        return ProbeResult(
            "missing", "nvidia-smi not on PATH",
            fix_hint="install the NVIDIA driver (gpu bundles need one)",
        )
    except subprocess.TimeoutExpired:
        return ProbeResult("error", f"nvidia-smi timed out after {timeout:g}s")
    if proc.returncode != 0:
        snippet = (proc.stderr or proc.stdout).strip().splitlines()
        return ProbeResult(
            "missing", snippet[0] if snippet else
            f"nvidia-smi exited {proc.returncode}",
        )
    gpus = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if not gpus:
        return ProbeResult("missing", "nvidia-smi reports no GPUs")
    return ProbeResult("ok", "; ".join(gpus))


def _adc_path() -> Path:
    """Google Application Default Credentials file location."""
    cloudsdk = os.environ.get("CLOUDSDK_CONFIG", "").strip()
    base = Path(cloudsdk).expanduser() if cloudsdk else Path.home() / ".config" / "gcloud"
    return base / "application_default_credentials.json"


def _envstr(name: str) -> str:
    return os.environ.get(name, "").strip()


def probe_llm_providers(*, probe: bool = False) -> dict[str, ProbeResult]:
    """Credential presence (and, when ``probe=True``, a 1-token API ping)
    per LLM provider, as :mod:`gap.agent.llm` reads them.

    - ``openrouter`` (default provider): ``OPENROUTER_API_KEY``.
    - ``vertex``: native Google SDKs authenticate via Application Default
      Credentials (``$GOOGLE_APPLICATION_CREDENTIALS`` or the gcloud ADC
      file); project/region come from the llm config, not env vars.

    When ``probe`` is true, every provider that *looks* configured gets a
    1-token API call (5 s timeout). A static check is a poor proxy: a
    stale-but-present key still reports ``ok``; the probe path catches
    that.
    """
    results: dict[str, ProbeResult] = {}

    if _envstr("OPENROUTER_API_KEY"):
        results["openrouter"] = ProbeResult("ok", "OPENROUTER_API_KEY set")
    else:
        results["openrouter"] = ProbeResult(
            "missing", "OPENROUTER_API_KEY not set",
            fix_hint="export OPENROUTER_API_KEY=...",
        )

    gac = _envstr("GOOGLE_APPLICATION_CREDENTIALS")
    if gac:
        if Path(gac).expanduser().is_file():
            results["vertex"] = ProbeResult(
                "ok", "ADC via $GOOGLE_APPLICATION_CREDENTIALS",
            )
        else:
            results["vertex"] = ProbeResult(
                "error",
                f"$GOOGLE_APPLICATION_CREDENTIALS points at a missing file: {gac}",
            )
    elif _adc_path().is_file():
        results["vertex"] = ProbeResult("ok", f"gcloud ADC ({_adc_path()})")
    else:
        results["vertex"] = ProbeResult(
            "missing", "no Application Default Credentials",
            fix_hint="gcloud auth application-default login",
        )
    if results["vertex"].ok:
        import importlib.util

        if importlib.util.find_spec("google") is None or (
            importlib.util.find_spec("google.genai") is None
        ):
            # Credentials alone are not enough: codegen on vertex imports
            # google-genai in *this* interpreter, so a missing extra means
            # `gap generate --provider vertex` cannot work. Downgrade the
            # probe (the pretty printer only surfaces hints on non-ok
            # probes, so an appended detail on an "ok" result is invisible).
            results["vertex"] = ProbeResult(
                "missing",
                results["vertex"].detail + "; google-genai not installed",
                fix_hint="uv sync --extra vertex  (or pip install "
                         "'graph-as-policy[vertex]')",
            )

    if probe:
        for name, res in results.items():
            if not res.ok:
                continue
            if name == "vertex" and not _envstr("GOOGLE_CLOUD_PROJECT"):
                # A ping without a project can only fail — and its error
                # ("re-check credentials") would point at the wrong culprit.
                results[name] = ProbeResult(
                    "missing",
                    f"{res.detail}; no GCP project set — ping skipped",
                    fix_hint="export GOOGLE_CLOUD_PROJECT=<project-id>",
                )
                continue
            ping = _ping_provider(
                name,
                model=_envstr("GAP_LLM_MODEL"),
                project_id=_envstr("GOOGLE_CLOUD_PROJECT"),
                region=(
                    _envstr("GOOGLE_CLOUD_REGION")
                    or _envstr("GOOGLE_CLOUD_LOCATION")
                    or "global"
                ),
            )
            # Preserve the cred-source detail; append the ping verdict.
            results[name] = ProbeResult(
                ping.status,
                detail=(f"{res.detail}; {ping.detail}" if res.detail else ping.detail),
                fix_hint=ping.fix_hint,
            )
    return results


@dataclass
class ResolvedVlmEnv:
    """The post-inheritance VLM config the vlm tool bundle will actually
    use, mirroring ``open-robot-skills/tools/vlm/tools.py`` resolvers.
    Surfaced in ``gap check`` so misconfigs (e.g. VLM defaulting to
    openrouter in a vertex-configured shell) are obvious before a run."""

    provider: str
    model: str
    vertex_project: str
    vertex_region: str
    provider_source: str
    """``"explicit"`` (``GAP_VLM_PROVIDER`` set), ``"inherited"`` (from
    ``GAP_LLM_PROVIDER``), or ``"default"`` (fell through to openrouter)."""


def resolve_vlm_env() -> ResolvedVlmEnv:
    """Mirror the vlm tool bundle's resolution chain (see
    ``open-robot-skills/tools/vlm/tools.py``). Kept in gap so ``gap check``
    can report what the bundle would resolve to without spawning the
    bundle subprocess."""
    if _envstr("GAP_VLM_PROVIDER"):
        provider, source = _envstr("GAP_VLM_PROVIDER").lower(), "explicit"
    elif _envstr("GAP_LLM_PROVIDER"):
        provider, source = _envstr("GAP_LLM_PROVIDER").lower(), "inherited"
    else:
        provider, source = "openrouter", "default"
    model = _envstr("GAP_VLM_MODEL") or _envstr("GAP_LLM_MODEL")
    project = (
        _envstr("GAP_VLM_PROJECT_ID")
        or _envstr("GOOGLE_CLOUD_PROJECT")
    )
    region = (
        _envstr("GAP_VLM_REGION")
        or _envstr("GOOGLE_CLOUD_REGION")
        or _envstr("GOOGLE_CLOUD_LOCATION")
        or "global"
    )
    return ResolvedVlmEnv(provider, model, project, region, source)


def probe_vlm_providers(
    *, probe: bool = False, registry_set: RegistrySet | None = None,
) -> dict[str, ProbeResult]:
    """Credential presence (and, when ``probe=True``, a 1-token API ping)
    for the resolved VLM provider.

    Unlike :func:`probe_llm_providers` (which reports ALL providers' cred
    presence so the user can pick), the VLM bundle dispatches to ONE
    provider per run — the one :func:`resolve_vlm_env` settles on. The
    report key is that resolved provider name; the detail string includes
    the resolution source (explicit / inherited / default) and resolved
    model/project/region so misconfigs (e.g. the VLM bundle silently
    falling through to the openrouter default with no API key) are
    visible before a run."""
    env = resolve_vlm_env()

    if env.provider == "openrouter":
        if _envstr("OPENROUTER_API_KEY") or _envstr("GAP_VLM_API_KEY"):
            base_url = _envstr("GAP_VLM_BASE_URL") or "https://openrouter.ai/api/v1"
            res = ProbeResult(
                "ok",
                f"base_url={base_url}"
                + (f", model={env.model}" if env.model else " (default model)"),
            )
        else:
            res = ProbeResult(
                "missing", "no OpenRouter API key",
                fix_hint=(
                    "export OPENROUTER_API_KEY=... (or GAP_VLM_API_KEY), or "
                    "route the vlm bundle elsewhere (export "
                    "GAP_VLM_PROVIDER=vertex or set GAP_LLM_PROVIDER — VLM "
                    "inherits when unset)"
                ),
            )
    elif env.provider == "vertex":
        if not env.vertex_project:
            res = ProbeResult(
                "missing", "no vertex project resolved",
                fix_hint=(
                    "export GAP_VLM_PROJECT_ID=..., or GOOGLE_CLOUD_PROJECT "
                    "(VLM inherits when unset)"
                ),
            )
        elif _adc_path().is_file() or _envstr("GOOGLE_APPLICATION_CREDENTIALS"):
            res = ProbeResult(
                "ok",
                f"project={env.vertex_project}, region={env.vertex_region}"
                + (f", model={env.model}" if env.model else " (default model)"),
            )
        else:
            res = ProbeResult(
                "missing", "no Application Default Credentials",
                fix_hint="gcloud auth application-default login",
            )
    else:
        res = ProbeResult(
            "error", f"unknown provider {env.provider!r}",
            fix_hint="set GAP_VLM_PROVIDER to openrouter | vertex",
        )

    # Prefix the detail with the resolution source so the misconfig
    # narrative ("vlm fell through to openrouter by default") is loud.
    src_label = {
        "explicit": "from GAP_VLM_PROVIDER",
        "inherited": "from GAP_LLM_PROVIDER",
        "default": "default (no GAP_VLM_PROVIDER or GAP_LLM_PROVIDER set)",
    }[env.provider_source]
    res = ProbeResult(
        res.status,
        detail=f"{src_label}" + (f" — {res.detail}" if res.detail else ""),
        fix_hint=res.fix_hint,
    )

    if probe and res.ok:
        ping = _ping_provider(
            env.provider, model=env.model,
            project_id=env.vertex_project, region=env.vertex_region,
        )
        if (
            env.provider == "vertex"
            and ping.status == "error"
            and _VERTEX_SDK_MISSING in ping.detail
        ):
            # The VLM never runs in the engine venv — the vlm bundle is an
            # out-of-process server with its own venv (which always carries
            # google-genai). Re-run the ping through that interpreter so a
            # missing engine-side vertex extra doesn't masquerade as a
            # credential failure.
            ping = _ping_vertex_via_vlm_bundle(env, registry_set)
        res = ProbeResult(
            ping.status,
            detail=f"{res.detail}; {ping.detail}" if res.detail else ping.detail,
            fix_hint=ping.fix_hint,
        )
    return {env.provider: res}


def _ping_vertex_via_vlm_bundle(
    env: ResolvedVlmEnv, registry_set: RegistrySet | None,
    *, timeout_s: float = 60.0,
) -> ProbeResult:
    """1-token vertex ping executed by the vlm bundle's venv interpreter —
    the environment that actually serves VLM calls at runtime."""
    venv_python: Path | None = None
    for spec in registry_set or []:
        cand = Path(spec.path) / "tools" / "vlm" / ".venv" / "bin" / "python"
        if cand.is_file():
            venv_python = cand
            break
    if venv_python is None:
        return ProbeResult(
            "unknown",
            "google-genai is not in the engine venv and no installed vlm "
            "bundle venv was found to probe instead; VLM calls run in the "
            "bundle venv, engine-side generation needs the vertex extra",
            fix_hint=(
                "uv run gap skills install vlm  ·  uv sync --extra vertex"
            ),
        )
    model = env.model or "gemini-2.5-flash"
    snippet = (
        "import os\n"
        "from google import genai\n"
        "from google.genai import types\n"
        "client = genai.Client(vertexai=True,"
        " project=os.environ['_GAP_PING_PROJECT'],"
        " location=os.environ['_GAP_PING_REGION'])\n"
        "client.models.generate_content(model=os.environ['_GAP_PING_MODEL'],"
        " contents='x',"
        " config=types.GenerateContentConfig(max_output_tokens=1))\n"
        "print('PING_OK')\n"
    )
    ping_env = dict(os.environ)
    ping_env.update({
        "_GAP_PING_PROJECT": env.vertex_project,
        "_GAP_PING_REGION": env.vertex_region,
        "_GAP_PING_MODEL": model,
    })
    try:
        proc = subprocess.run(
            [str(venv_python), "-c", snippet],
            capture_output=True, text=True, timeout=timeout_s, env=ping_env,
        )
    except subprocess.TimeoutExpired:
        return ProbeResult(
            "error", f"vlm bundle ping timed out after {timeout_s:g}s",
        )
    if proc.returncode == 0 and "PING_OK" in proc.stdout:
        return ProbeResult(
            "ok",
            f"ping ok via vlm bundle venv (project={env.vertex_project}, "
            f"region={env.vertex_region}, model={model})",
        )
    err_lines = [
        ln for ln in (proc.stderr or proc.stdout).strip().splitlines() if ln
    ]
    msg = (err_lines[-1] if err_lines else f"exit {proc.returncode}")[:200]
    return ProbeResult(
        "error", f"ping failed (via vlm bundle venv): {msg}",
        fix_hint="re-check credentials and model name; see provider docs",
    )


#: Sentinel detail for "the google-genai SDK is not importable in the venv
#: running this probe" — distinguishes a local install gap from a real
#: credential/model failure (the fix hints differ completely).
_VERTEX_SDK_MISSING = "google-genai not installed in this environment"


def _ping_provider(
    provider: str, *, model: str, project_id: str, region: str,
    timeout_s: float = 5.0,
) -> ProbeResult:
    """1-token API call to ``provider`` — returns ``ok`` on a 2xx, ``error``
    with the exception string on anything else. Imports the SDK lazily
    so a non-installed provider degrades to a clear error, not an import
    error at module load."""
    try:
        if provider == "openrouter":
            import httpx
            base_url = _envstr("GAP_VLM_BASE_URL") or _envstr("GAP_LLM_ENDPOINT") \
                or "https://openrouter.ai/api/v1"
            key = _envstr("OPENROUTER_API_KEY") or _envstr("GAP_VLM_API_KEY")
            headers = {"Authorization": f"Bearer {key}"} if key else {}
            ping_model = model or "gemini-3.1-flash-lite-preview"
            with httpx.Client(timeout=timeout_s) as c:
                r = c.post(
                    base_url.rstrip("/") + "/chat/completions",
                    json={
                        "model": ping_model,
                        "messages": [{"role": "user", "content": "x"}],
                        "max_tokens": 1,
                    },
                    headers=headers,
                )
                r.raise_for_status()
            return ProbeResult("ok", f"ping ok (model={ping_model})")
        if provider == "vertex":
            try:
                from google import genai
                from google.genai import types
            except ImportError:
                return ProbeResult(
                    "error", _VERTEX_SDK_MISSING,
                    fix_hint=(
                        "install the vertex extra: uv sync --extra vertex "
                        "(or pip install 'graph-as-policy[vertex]')"
                    ),
                )
            client = genai.Client(
                vertexai=True, project=project_id, location=region,
            )
            client.models.generate_content(
                model=model or "gemini-2.5-flash",
                contents="x",
                config=types.GenerateContentConfig(max_output_tokens=1),
            )
            return ProbeResult(
                "ok",
                f"ping ok (project={project_id}, region={region}, "
                f"model={model or 'gemini-2.5-flash'})",
            )
    except Exception as exc:  # noqa: BLE001 — probe surface, all errors equal
        msg = str(exc).splitlines()[0][:200]
        return ProbeResult(
            "error", f"ping failed: {msg}",
            fix_hint="re-check credentials and model name; see provider docs",
        )
    return ProbeResult("error", f"unknown provider {provider!r}")


@dataclass
class EnvironmentReport:
    python_version: str
    gap_version: str
    gpu: ProbeResult
    llm_providers: dict[str, ProbeResult]
    vlm_provider: dict[str, ProbeResult] = field(default_factory=dict)
    """Single-entry ``{resolved_provider: ProbeResult}`` — the VLM bundle
    dispatches to one provider per run (see :func:`resolve_vlm_env`)."""

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "python_version": self.python_version,
            "gap_version": self.gap_version,
            "gpu": self.gpu.to_json_dict(),
            "llm_providers": {
                name: probe.to_json_dict()
                for name, probe in self.llm_providers.items()
            },
            "vlm_provider": {
                name: probe.to_json_dict()
                for name, probe in self.vlm_provider.items()
            },
        }


def probe_environment(
    *, probe: bool = False, registry_set: RegistrySet | None = None,
) -> EnvironmentReport:
    import gap

    return EnvironmentReport(
        python_version=platform.python_version(),
        gap_version=getattr(gap, "__version__", "unknown"),
        gpu=probe_gpu(),
        llm_providers=probe_llm_providers(probe=probe),
        vlm_provider=probe_vlm_providers(probe=probe, registry_set=registry_set),
    )


# ---------------------------------------------------------------------------
# Per-bundle probes
# ---------------------------------------------------------------------------


def dep_fix_hint(
    registry_root: Path,
    *,
    dist_name: str | None,
    bundle: str,
    has_extra: bool,
) -> str:
    """How to install a bundle's missing deps, for *this* registry."""
    if not has_extra:
        return (
            f"declare a {bundle!r} extra in {registry_root}/pyproject.toml "
            f"and install it"
        )
    if (registry_root / "uv.lock").is_file():
        return f"uv sync --extra {bundle}  (in {registry_root})"
    if dist_name:
        return f"pip install '{dist_name}[{bundle}]'"
    return f"install the {bundle!r} extra's dependencies (see {registry_root}/pyproject.toml)"


def probe_bundle_import(
    name: str,
    bundle_dir: Path,
    *,
    kind: Literal["tool", "skill", "policy"],
    fix_hint: str = "",
) -> tuple[ProbeResult, Any | None]:
    """Import-probe one bundle in isolation.

    Registers the bundle into a throwaway registry so one broken bundle
    never masks the rest. Returns the probe verdict and, on success, the
    :class:`~gap.skills._registry.SkillInfo` (whose ``tools_module`` /
    ``module`` host the optional ``weights_cached()`` / ``prefetch()``
    hooks).
    """
    from ._registry import SkillsRegistry

    reg = SkillsRegistry()
    try:
        reg.register_bundle(name, bundle_dir, kind=kind)
    except ImportError as exc:
        missing = getattr(exc, "name", None) or str(exc)
        return ProbeResult(
            "missing", f"missing {missing}", fix_hint=fix_hint,
        ), None
    except Exception as exc:
        return ProbeResult("error", f"import probe failed: {exc}"), None
    return ProbeResult("ok"), reg.get(name)


def _probe_requirements(
    meta: SkillMeta, *, gpu: ProbeResult,
) -> list[tuple[str, ProbeResult]]:
    """Evaluate the bundle's declared ``gap.requires`` block."""
    req = meta.requires
    if req is None:
        return []
    results: list[tuple[str, ProbeResult]] = []
    if req.gpu:
        results.append(("gpu", gpu))
    for var in req.env:
        if os.environ.get(var, "").strip():
            results.append((f"env:{var}", ProbeResult("ok")))
        else:
            results.append((f"env:{var}", ProbeResult(
                "missing", f"{var} not set",
                fix_hint=f"export {var}=...  (see the bundle's SKILL.md)",
            )))
    if req.env_any:
        set_vars = [v for v in req.env_any if os.environ.get(v, "").strip()]
        label = f"env_any:{'|'.join(req.env_any)}"
        if set_vars:
            results.append((label, ProbeResult("ok", f"{set_vars[0]} set")))
        else:
            results.append((label, ProbeResult(
                "missing", f"none of {', '.join(req.env_any)} set",
                fix_hint=(
                    f"export {req.env_any[0]}=...  (any one of these works; "
                    f"see the bundle's SKILL.md)"
                ),
            )))
    return results


def _probe_weights(meta: SkillMeta, info: Any | None) -> ProbeResult:
    """The optional filesystem-only ``weights_cached()`` hook."""
    req = meta.requires
    if req is None or not req.weights:
        return ProbeResult("ok", "no weights declared")
    if info is None:
        return ProbeResult(
            "unknown", "deps not importable",
            fix_hint="run `gap skills check --download` after installing deps",
        )
    hook = None
    for module in (getattr(info, "tools_module", None), getattr(info, "module", None)):
        fn = getattr(module, "weights_cached", None) if module is not None else None
        if callable(fn):
            hook = fn
            break
    if hook is None:
        return ProbeResult(
            "unknown", "bundle has no weights_cached() hook",
            fix_hint="run `gap skills check --download` to (pre)fetch",
        )
    try:
        cached = hook()
    except Exception as exc:
        return ProbeResult("error", f"weights_cached() failed: {exc}")
    if cached is True:
        return ProbeResult("ok", "cached")
    if cached is False:
        return ProbeResult(
            "missing", "not cached (downloads on first use)",
            fix_hint="gap skills check --download",
        )
    return ProbeResult(
        "unknown", "weights_cached() returned None",
        fix_hint="run `gap skills check --download` to (pre)fetch",
    )


@dataclass
class BundleCapability:
    """Operational status of one bundle in one registry."""

    name: str
    kind: Literal["tool", "skill", "policy"]
    registry: str
    bundle_dir: Path
    deps: ProbeResult
    requirements: list[tuple[str, ProbeResult]] = field(default_factory=list)
    weights: ProbeResult = field(default_factory=lambda: ProbeResult("ok"))
    shadowed_by: str = ""
    """Set when a higher-precedence registry already claims this bundle
    name — the bundle is inert (never imported, never dispatched)."""

    @property
    def status(self) -> Literal["ready", "not-ready", "shadowed"]:
        if self.shadowed_by:
            return "shadowed"
        if not self.deps.ok:
            return "not-ready"
        if any(not probe.ok for _, probe in self.requirements):
            return "not-ready"
        return "ready"

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "registry": self.registry,
            "bundle_dir": str(self.bundle_dir),
            "status": self.status,
            "deps": self.deps.to_json_dict(),
            "requirements": {
                label: probe.to_json_dict()
                for label, probe in self.requirements
            },
            "weights": self.weights.to_json_dict(),
            **({"shadowed_by": self.shadowed_by} if self.shadowed_by else {}),
        }


@dataclass
class SkillRunnability:
    """A skill bundle's rollup: blocked exactly by not-ready tool bundles."""

    name: str
    registry: str
    tools: dict[str, str] = field(default_factory=dict)
    """``{allowed_tool_name: owner}`` where owner is a bundle name,
    ``"connector"`` (robot.*/sim.* — satisfied at run time), or
    ``"unknown"`` (unresolvable prefix — flagged)."""
    blocked_by: list[str] = field(default_factory=list)
    unknown_tools: list[str] = field(default_factory=list)
    self_ready: bool = True
    """The skill bundle's own deps/requirements verdict."""

    @property
    def status(self) -> Literal["ready", "blocked"]:
        return "ready" if self.self_ready and not self.blocked_by else "blocked"

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "registry": self.registry,
            "status": self.status,
            "tools": dict(self.tools),
            "blocked_by": list(self.blocked_by),
            "unknown_tools": list(self.unknown_tools),
        }


@dataclass
class CheckReport:
    """Everything ``gap check`` reports; ``to_json_dict()`` is the stable
    ``--format json`` schema (bump ``schema_version`` on breaking change)."""

    environment: EnvironmentReport
    registries: list[RegistrySpec] = field(default_factory=list)
    bundles: list[BundleCapability] = field(default_factory=list)
    skills: list[SkillRunnability] = field(default_factory=list)
    schema_version: int = 1

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "environment": self.environment.to_json_dict(),
            "registries": [
                {
                    "name": spec.name,
                    "path": str(spec.path),
                    "source": spec.source,
                    "dist_name": spec.dist_name,
                    "origin": spec.origin,
                }
                for spec in self.registries
            ],
            "bundles": [b.to_json_dict() for b in self.bundles],
            "skills": [s.to_json_dict() for s in self.skills],
        }


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


def _tool_owner(tool_name: str, owner_by_prefix: dict[str, str]) -> str:
    prefix = tool_name.split(".", 1)[0]
    if prefix in CONNECTOR_PREFIXES:
        return "connector"
    return owner_by_prefix.get(prefix, "unknown")


def build_check_report(
    registry_set: RegistrySet, *, probe: bool = False,
) -> CheckReport:
    """Probe every bundle of every active registry and roll up skills.

    ``probe=True`` makes :func:`probe_llm_providers` /
    :func:`probe_vlm_providers` issue a real 1-token API call to each
    configured provider; default is static (env-var / ADC presence
    only). Bundle import + weights probes are always run regardless.
    """
    environment = probe_environment(probe=probe, registry_set=registry_set)
    gpu = environment.gpu

    bundles: list[BundleCapability] = []
    metas: dict[str, SkillMeta] = {}          # winner per bundle name
    claimed: dict[str, str] = {}              # bundle name -> registry name

    for spec in registry_set:
        extras = load_checkout_extras(spec.path) or {}
        for report in validate_checkout(spec.path):
            shadowed_by = claimed.get(report.name, "")
            if not shadowed_by:
                claimed[report.name] = spec.name

            if report.meta is None:
                bundles.append(BundleCapability(
                    name=report.name, kind=report.kind, registry=spec.name,
                    bundle_dir=report.bundle_dir,
                    deps=ProbeResult("error", "SKILL.md rejected — run `gap skills check`"),
                    shadowed_by=shadowed_by,
                ))
                continue

            if shadowed_by:
                # Inert: never imported (the synthetic gap_skills.* namespace
                # belongs to the winner), so no probes either.
                bundles.append(BundleCapability(
                    name=report.name, kind=report.kind, registry=spec.name,
                    bundle_dir=report.bundle_dir, deps=ProbeResult("ok"),
                    shadowed_by=shadowed_by,
                ))
                continue

            metas[report.name] = report.meta
            hint = dep_fix_hint(
                spec.path, dist_name=spec.dist_name, bundle=report.name,
                has_extra=report.name in extras,
            )
            deps, info = probe_bundle_import(
                report.name, report.bundle_dir, kind=report.kind,
                fix_hint=hint,
            )
            bundles.append(BundleCapability(
                name=report.name, kind=report.kind, registry=spec.name,
                bundle_dir=report.bundle_dir, deps=deps,
                requirements=_probe_requirements(report.meta, gpu=gpu),
                weights=_probe_weights(report.meta, info),
            ))

    # Skill rollup — owners resolved statically from declared gap.tools
    # names (works even when a bundle's deps are not installed).
    owner_by_prefix: dict[str, str] = {}
    for name, meta in metas.items():
        for tool_name in meta.tools:
            owner_by_prefix.setdefault(tool_name.split(".", 1)[0], name)

    capability_by_name = {
        b.name: b for b in bundles if not b.shadowed_by
    }
    skills: list[SkillRunnability] = []
    for bundle in bundles:
        if bundle.kind != "skill" or bundle.shadowed_by:
            continue
        meta = metas.get(bundle.name)
        if meta is None:
            continue
        tools: dict[str, str] = {}
        blocked: list[str] = []
        unknown: list[str] = []
        for tool_name in meta.allowed_tools:
            owner = _tool_owner(tool_name, owner_by_prefix)
            tools[tool_name] = owner
            if owner == "unknown":
                unknown.append(tool_name)
            elif owner not in ("connector",):
                owner_cap = capability_by_name.get(owner)
                if owner_cap is not None and owner_cap.status == "not-ready":
                    if owner not in blocked:
                        blocked.append(owner)
        skills.append(SkillRunnability(
            name=bundle.name, registry=bundle.registry, tools=tools,
            blocked_by=blocked, unknown_tools=unknown,
            self_ready=bundle.status == "ready",
        ))

    return CheckReport(
        environment=environment,
        registries=list(registry_set),
        bundles=bundles,
        skills=skills,
    )
