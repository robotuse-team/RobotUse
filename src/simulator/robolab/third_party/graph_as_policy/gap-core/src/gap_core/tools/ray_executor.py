"""Optional Ray-backed shared skill actors for parallel benchmark workers.

Each parallel benchmark worker normally loads its own copy of the heavy
perception models (sam3 + dino ≈ a few GB of VRAM per worker). With many
workers per GPU, sharing one model instance per bundle saves the
difference. :class:`RayToolExecutor` wraps a heavy tool bundle's
callables in a single ``@ray.remote`` actor (one actor per bundle,
created lazily on first invoke) and :func:`substitute_ray_tools`
re-routes the matching tools' dispatch inside a
:class:`gap.tools.ToolRegistry` through those actors.

Ray is an optional extra::

    pip install "graph-as-policy[ray]"

The module imports without ray installed; only constructing a
:class:`RayToolExecutor` without injecting a ray module raises. Tests
inject a stub ray module — no cluster needed.
"""

from __future__ import annotations

import functools
import logging
import threading
from collections.abc import Callable, Iterable
from typing import Any

logger = logging.getLogger(__name__)

#: Synthetic-package prefix bundle tool modules import under (see
#: gap.skills._registry). ``gap_skills.tools.<bundle>.tools`` →
#: bundle name is the third dotted segment.
_BUNDLE_MODULE_PREFIX = "gap_skills.tools."


def _import_ray() -> Any:
    """Guarded ray import with an actionable error message."""
    try:
        import ray
    except ImportError as e:  # pragma: no cover - exercised via stub tests
        raise ImportError(
            "ray is not installed — the shared-skill-actor path needs the "
            "[ray] extra: pip install 'graph-as-policy[ray]'"
        ) from e
    return ray


def bundle_for_tool(descriptor: Any) -> str | None:
    """The owning bundle name for a registry tool, or None.

    Resolved from the descriptor's ``metadata["module"]`` — bundle tool
    modules always live under the ``gap_skills.tools.<bundle>``
    synthetic package; connector tools (``robot.*`` / ``sim.*``) and
    in-repo ``@tool`` plugins don't and return None.
    """
    module = str((getattr(descriptor, "metadata", None) or {}).get("module", ""))
    if not module.startswith(_BUNDLE_MODULE_PREFIX):
        return None
    rest = module[len(_BUNDLE_MODULE_PREFIX):]
    bundle = rest.split(".", 1)[0]
    return bundle or None


class _BundleActor:
    """Actor body hosting one bundle's tool callables.

    Instantiated remotely by ray (or in-process by the test stub). The
    callables map is captured at actor-construction time, so the heavy
    model state each callable closes over is loaded exactly once in the
    actor process.
    """

    def __init__(self, callables: dict[str, Callable[..., Any]]) -> None:
        self._callables = dict(callables)

    def invoke(self, tool_name: str, kwargs: dict[str, Any]) -> Any:
        fn = self._callables.get(tool_name)
        if fn is None:
            raise KeyError(
                f"tool {tool_name!r} not hosted by this bundle actor "
                f"(hosted: {sorted(self._callables)})"
            )
        return fn(**kwargs)


class RayToolExecutor:
    """Lazily creates one shared ray actor per heavy tool bundle.

    Args:
        ray_module: Injection seam for tests (a stub exposing
            ``remote(cls)``, ``get(ref)`` and optionally ``init``).
            ``None`` imports the real ray (raises without the extra).
        actor_options: Extra options forwarded to ``Actor.options(...)``
            (e.g. ``{"num_gpus": 0.25}``). Empty → defaults.
    """

    def __init__(
        self,
        *,
        ray_module: Any = None,
        actor_options: dict[str, Any] | None = None,
    ) -> None:
        self._ray = ray_module if ray_module is not None else _import_ray()
        self._actor_options = dict(actor_options or {})
        self._actors: dict[str, Any] = {}
        self._pending: dict[str, dict[str, Callable[..., Any]]] = {}
        self._lock = threading.Lock()

    def register_bundle(
        self, bundle: str, callables: dict[str, Callable[..., Any]],
    ) -> None:
        """Stage a bundle's callables; the actor spawns on first invoke."""
        with self._lock:
            if bundle in self._actors:
                raise ValueError(f"bundle {bundle!r} actor already created")
            self._pending.setdefault(bundle, {}).update(callables)

    def knows_bundle(self, bundle: str) -> bool:
        """True when the bundle is staged or its actor is already live."""
        with self._lock:
            return bundle in self._actors or bundle in self._pending

    def _actor_for(self, bundle: str) -> Any:
        with self._lock:
            actor = self._actors.get(bundle)
            if actor is not None:
                return actor
            callables = self._pending.pop(bundle, None)
            if callables is None:
                raise KeyError(
                    f"no callables registered for bundle {bundle!r} "
                    f"(registered: {sorted(self._actors) + sorted(self._pending)})"
                )
            remote_cls = self._ray.remote(_BundleActor)
            if self._actor_options:
                remote_cls = remote_cls.options(**self._actor_options)
            actor = remote_cls.remote(callables)
            self._actors[bundle] = actor
            logger.info(
                "RayToolExecutor: spawned shared actor for bundle %r "
                "(%d tool(s))", bundle, len(callables),
            )
            return actor

    def invoke(self, bundle: str, tool_name: str, /, **kwargs: Any) -> Any:
        """Invoke ``tool_name`` on the bundle's shared actor (blocking)."""
        actor = self._actor_for(bundle)
        return self._ray.get(actor.invoke.remote(tool_name, kwargs))

    @property
    def live_bundles(self) -> list[str]:
        """Bundles whose actor has been created."""
        with self._lock:
            return sorted(self._actors)


def substitute_ray_tools(
    registry: Any,
    bundles: Iterable[str] | None = None,
    *,
    executor: RayToolExecutor | None = None,
    ray_module: Any = None,
) -> RayToolExecutor:
    """Re-route heavy bundle tools in *registry* through shared ray actors.

    For every python-transport tool whose module lives under
    ``gap_skills.tools.<bundle>`` (and, when *bundles* is given, whose
    bundle is listed), the registry's in-process dispatch callable is
    replaced by a proxy that calls the bundle's shared actor. Tools that
    require ctx injection are left in-process (a NodeContext is not
    picklable).

    Returns the executor (created here unless injected) so the caller
    can inspect / reuse it across registries.
    """
    if executor is None:
        executor = RayToolExecutor(ray_module=ray_module)

    adapter = registry.python_adapter
    staged: dict[str, dict[str, Callable[..., Any]]] = {}
    proxies: list[tuple[str, str, Callable[..., Any], bool]] = []
    allow = set(bundles) if bundles is not None else None

    for name, desc in registry.by_transport("python").items():
        bundle = bundle_for_tool(desc)
        if bundle is None or (allow is not None and bundle not in allow):
            continue
        if (desc.metadata or {}).get("requires_ctx"):
            logger.debug(
                "substitute_ray_tools: %r requires ctx — left in-process", name,
            )
            continue
        fn = adapter._callables.get(name)
        if fn is None:
            continue
        staged.setdefault(bundle, {})[name] = fn
        proxies.append((bundle, name, fn, False))

    for bundle, callables in staged.items():
        # A shared executor across worker registries: the first caller
        # stages (and eventually spawns) the bundle actor; later
        # registries only swap their dispatch onto the existing actor.
        if bundle in executor.live_bundles:
            continue
        executor.register_bundle(bundle, callables)

    for bundle, name, fn, requires_ctx in proxies:
        @functools.wraps(fn)
        def _proxy(
            *args: Any,
            _bundle: str = bundle,
            _name: str = name,
            **kwargs: Any,
        ) -> Any:
            if args:
                raise TypeError(
                    f"ray-routed tool {_name!r} accepts keyword arguments only"
                )
            return executor.invoke(_bundle, _name, **kwargs)

        adapter.register(name, _proxy, requires_ctx=requires_ctx)
        logger.debug(
            "substitute_ray_tools: %r now dispatches via bundle actor %r",
            name, bundle,
        )

    return executor


__all__ = [
    "RayToolExecutor",
    "bundle_for_tool",
    "substitute_ray_tools",
]
