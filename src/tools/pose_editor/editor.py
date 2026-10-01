"""Per-candidate pose edit panels with step and cumulative translation/rotation.

Each panel distinguishes edits of the generated draft from continued edits of
an accepted revision, and shows the remaining budget when one is configured.
"""
from __future__ import annotations

GRASP_AXES = ('dx_mm', 'dy_mm', 'dz_mm')
PLACE_AXES = ('dx_m', 'dy_m', 'dz_m')
AXES_DEG = ('roll_deg', 'pitch_deg', 'yaw_deg')
HISTORY_STEPS = 6


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.
    return number if number == number and abs(number) != float('inf') else 0.


def _axes(arguments, names, scale=1.):
    return {name: round(_number(arguments.get(name)) * scale, 2) for name in names}


def _nonzero(values):
    return {name: value for name, value in values.items() if value}


class PoseEditor:
    """Per-candidate edit history, carried forward as each edit makes a new ref."""

    def __init__(self, limit, *, axes=GRASP_AXES, millimetre_scale=1., translation_frame=None):
        self.limit = limit
        self.axes = tuple(axes)
        self.scale = millimetre_scale
        self.translation_frame = translation_frame or ('local' if self.axes == GRASP_AXES else 'base')
        if self.translation_frame not in ('local', 'base'):
            raise ValueError('translation_frame must be local or base')
        self.output_axes = tuple(name[:-2] + '_mm' if name.endswith('_m') and self.scale == 1000. else name
                                 for name in self.axes)
        self._history = {}

    def reset(self):
        self._history = {}

    def record(self, origin_ref, new_ref, arguments, *, used):
        """Log one edit and return the panel the worker should read."""
        translations = _axes(arguments, self.axes, self.scale)
        applied = {**dict(zip(self.output_axes, translations.values())), **_axes(arguments, AXES_DEG)}
        # Editing the generator's draft again restarts the chain; editing the ref
        # a previous edit returned continues it. The panel identifies both cases.
        restarted = origin_ref not in self._history
        steps = list(self._history.get(origin_ref, ()))
        steps.append(applied)
        self._history[new_ref] = steps
        cumulative = {name: round(sum(step.get(name, 0.) for step in steps), 2)
                      for name in (*self.output_axes, *AXES_DEG)}
        panel = {
            'step': len(steps),
            'limit': self.limit,
            'remaining': None if self.limit is None else max(0, self.limit - used),
            'applied': _nonzero(applied) or 'nothing (all axes zero)',
            'cumulative_from_original': _nonzero(cumulative) or 'none; this draft still sits where '
                                                                'the generator put it',
            'editing': ('the generator draft, so this chain starts over' if restarted
                        else 'the draft your last edit produced, so offsets accumulate'),
            'reading': 'These are offsets from the generated draft, not from the object. '
                       + ('Grasp translations use local gripper axes; dz follows the approach axis. '
                        if self.translation_frame == 'local' else 'Translations use robot-base XYZ; dz follows base Z. ')
                       + 'Read the render for this step before deciding the next one.',
        }
        if len(steps) > 1:
            panel['history'] = [_nonzero(step) or 'no-op' for step in steps[-HISTORY_STEPS:]]
            repeats = [step for step in steps if step == applied]
            if len(repeats) > 2:
                panel['warning'] = (f'This is the same edit {len(repeats)} times. Repeating one '
                                    f'axis has moved the draft '
                                    f'{cumulative.get(self.output_axes[2], 0)} along z in total. '
                                    f'If the render does not look closer, '
                                    f'change axis or rotate instead of repeating.')
        return panel

    def cumulative(self, ref):
        steps = self._history.get(ref, ())
        return {name: round(sum(step.get(name, 0.) for step in steps), 2)
                for name in (*self.output_axes, *AXES_DEG)}
