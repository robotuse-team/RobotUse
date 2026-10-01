"""Read native task progress without changing simulation or success criteria."""
from __future__ import annotations

import math
from numbers import Real


def task_score(connector, *, enabled: bool) -> dict:
    """Keep the normalized subtask score separate from the environment reward."""
    result = dict(score_enabled=enabled, score_evaluated=False, score=None,
                  score_source=None, score_error=None)
    if not enabled:
        return result
    try:
        from robolab.core.events.subtask_recorder import SubtaskCompletionRecorderTerm

        native = connector.env._env
        manager = native.recorder_manager
        term = manager.get_term(SubtaskCompletionRecorderTerm) if manager is not None else None
        if term is None or not term.subtask_state_machines:
            raise ValueError('Native subtask scoring is unavailable for this task')
        # Recorder infos can retain the preceding episode until the first step
        # after reset. The live state machine resets immediately; this getter
        # reads its current progress without stepping conditions or physics.
        state = term.subtask_state_machines[0].get_subtask_state()
        value = state['score']
        if state['total'] <= 0:
            raise ValueError('Native task has no scored subtasks')
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Native subtask score must be a finite number between 0 and 1')
        result.update(score_evaluated=True, score=float(value),
                      score_source='robolab.SubtaskStateMachine.get_subtask_state')
    except Exception as exc:
        result['score_error'] = f'{type(exc).__name__}: {exc}'
    return result
