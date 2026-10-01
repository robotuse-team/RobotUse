"""Standalone Isaac CLI exit after results, environment close and worker cleanup.

Kit plugin/CPython finalization can hang or segfault at process shutdown.
Its default fast_shutdown instead exits before callers write results.
Own the process only at CLI boundaries: complete all application work first,
preserve its exit status, flush output, then skip Kit/CPython native finalizers.
Library calls retain normal Python lifetime behavior.
"""
import os
import sys
import traceback

OWNS_PROCESS = False


def run_native_cli(main):
    global OWNS_PROCESS
    OWNS_PROCESS = True
    status = 1
    try:
        status = int(main() or 0)
    except KeyboardInterrupt:
        status = 130
    except SystemExit as exc:
        status = exc.code if isinstance(exc.code, int) else 0 if exc.code is None else 1
        if isinstance(exc.code, str):
            print(exc.code, file=sys.stderr)
    except BaseException:
        traceback.print_exc()
    try:
        from src.runtime.worker import close_workers
        close_workers()
    except Exception:
        traceback.print_exc()
        status = status or 1
    try:
        # Optional GPU planners own a different Warp interpreter. Only close
        # the module if used; default native execution starts no such worker.
        curobo_process = sys.modules.get('src.tools.curobo.process')
        if curobo_process is not None:
            curobo_process.close_workers()
    except Exception:
        traceback.print_exc()
        status = status or 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(status)
