"""``gap policy`` subcommand — serve / list learned-policy bundles.

Policies are ``kind='policy'`` skill bundles (under ``<registry>/policies/``)
that own their server launch recipe via SKILL.md ``gap.serving:``. The CLI
discovers them through the active registry set (same precedence as
``gap skills list``) and spawns them via :class:`PolicyManager`, which runs
the bundle's own venv via ``uv run --project <bundle_dir>``.
"""

from __future__ import annotations

import argparse
from typing import Any


def register(subparsers: argparse._SubParsersAction) -> None:
    sp = subparsers.add_parser(
        "policy",
        help="Manage learned-policy servers (serve a bundle, list known bundles)",
    )
    psub = sp.add_subparsers(dest="policy_command")

    serve = psub.add_parser(
        "serve",
        help="Spawn a policy server from a bundle and block until Ctrl-C",
    )
    serve.add_argument(
        "bundle",
        help="Policy bundle name (see `gap policy list`), e.g. pi05-libero",
    )
    serve.add_argument(
        "--port", type=int, default=None,
        help="Serve on this port (default: an OS-allocated free port)",
    )
    serve.add_argument(
        "--startup-timeout", type=float, default=900.0, metavar="SECS",
        help="How long to wait for the server port to open (default 900; "
             "first run downloads checkpoints)",
    )
    _add_registry_args(serve)
    serve.set_defaults(func=_handle_serve)

    lst = psub.add_parser(
        "list", help="List the policy bundles discovered in active registries",
    )
    _add_registry_args(lst)
    lst.set_defaults(func=_handle_list)

    sp.set_defaults(func=_handle_help, _parser=sp)


def _add_registry_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--skills", action="append", default=None, metavar="PATH",
        help="Registry checkout root(s); repeatable. Overrides "
             "$GAP_SKILLS_PATH and configured registries (default: the "
             "resolved registry set — see `gap registry list`)",
    )
    parser.add_argument(
        "--registry", default=None, metavar="NAME",
        help="Restrict to one active registry by name",
    )


def _handle_help(args: argparse.Namespace) -> int:
    args._parser.print_help()
    return 1


def _load_policy_bundles(args: argparse.Namespace) -> dict[str, Any]:
    """Return ``{bundle_name: SkillInfo}`` for every kind='policy' bundle
    in the active registry set."""
    from gap.skills import load_skills, resolve_registries
    from gap.skills.registries import RegistrySet

    registry_set = resolve_registries(args.skills, required=True)
    if getattr(args, "registry", None):
        registry_set = RegistrySet([registry_set.get(args.registry)])

    bundles: dict[str, Any] = {}
    for spec in registry_set:
        reg = load_skills(spec.path)
        for info in reg.list_skills(kind="policy"):
            # Higher-precedence registries win on name collision.
            bundles.setdefault(info.name, info)
    return bundles


def _handle_list(args: argparse.Namespace) -> int:
    try:
        bundles = _load_policy_bundles(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}")
        return 2

    if not bundles:
        print(
            "0 policy bundle(s) discovered.\n"
            "Add one under <registry>/policies/<name>/ with a SKILL.md that "
            "declares `gap.serving:`."
        )
        return 0

    print(f"{len(bundles)} policy bundle(s):\n")
    for name in sorted(bundles):
        info = bundles[name]
        serving = info.meta.serving
        print(f"  {name}")
        if serving:
            cmd = " ".join(serving.command)
            print(f"    command:  {cmd}")
            print(f"    protocol: {serving.protocol}")
            if serving.weights_uri:
                print(f"    weights:  {serving.weights_uri}")
        else:
            print("    (no gap.serving: block — `gap policy serve` will fail)")
        print()
    print("serve one with: gap policy serve <bundle> [--port N]")
    return 0


def _handle_serve(args: argparse.Namespace) -> int:
    import logging
    import time
    from contextlib import contextmanager

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    from gap.runtime.policy_manager import PolicyConfigError, PolicyManager

    try:
        bundles = _load_policy_bundles(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}")
        return 2

    if args.bundle not in bundles:
        available = ", ".join(sorted(bundles)) or "(none)"
        print(
            f"error: unknown policy bundle {args.bundle!r} "
            f"(available: {available})"
        )
        return 2

    info = bundles[args.bundle]
    serving = info.meta.serving
    if serving is None:
        print(
            f"error: policy bundle {args.bundle!r} declares no "
            f"`gap.serving:` block in SKILL.md"
        )
        return 2

    policy_id = args.bundle
    entries = {
        policy_id: {
            "command": list(serving.command),
            "bundle_dir": info.meta.bundle_dir,
            "env": dict(serving.env or {}),
        }
    }

    @contextmanager
    def _pinned_port(port: int | None):
        """Pin the manager's port allocation to a user-chosen port.

        PolicyManager always allocates a free port for the ``{port}``
        placeholder; for a long-lived `gap policy serve` the user wants
        a stable, well-known port, so we swap the allocator for the
        duration of the spawn.
        """
        if port is None:
            yield
            return
        import gap.runtime.policy_manager as pm_mod

        original = pm_mod._allocate_free_port
        pm_mod._allocate_free_port = lambda: port
        try:
            yield
        finally:
            pm_mod._allocate_free_port = original

    manager = PolicyManager(
        entries=entries, startup_timeout_s=float(args.startup_timeout),
    )
    try:
        with _pinned_port(args.port):
            manager.boot_all([policy_id])
    except PolicyConfigError as exc:
        print(f"FAIL: {exc}")
        return 2
    except Exception as exc:
        print(f"FAIL: {exc}")
        return 1

    url = manager.url_for(policy_id)
    print(f"policy {policy_id!r} serving on {url} — Ctrl-C to stop")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nshutting down ...")
    finally:
        manager.shutdown_all()
    return 0
