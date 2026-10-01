"""Simulation-clock budgets; planning/model latency never advances this clock."""
import math


class MotionBudgetExceeded(RuntimeError):
    def __init__(self, budget, required_steps):
        super().__init__('planned motion exceeds remaining simulation time')
        self.budget = dict(budget)
        self.required_steps = required_steps


def simulation_budget(connector):
    read = getattr(getattr(connector, 'env', None), 'simulation_budget', None)
    return read() if callable(read) else None


def gripper_settle_steps(connector, operation, fallback):
    env = getattr(connector, 'env', None)
    steps = getattr(env, f'gripper_{operation}_settle_steps', fallback)
    if type(steps) is not int or steps < 1:
        raise ValueError('positive integer gripper dwell required')
    return steps


def require_motion_budget(connector, segments, *, gripper_steps=0):
    """Reserve streamed samples, gripper dwell and nominal endpoint settling.

    Endpoint convergence can take longer; the native terminal guard remains
    authoritative. This estimate never changes a path or extends the horizon.
    """
    budget = simulation_budget(connector)
    if budget is None:
        return
    # The native streamer may retime samples. Read its configured scale without
    # changing it; gripper dwell and endpoint settling are not speed-scaled.
    speed = getattr(connector.env, 'motion_speed_scale', 1.)
    if not math.isfinite(speed) or speed <= 0:
        raise ValueError('invalid trajectory speed scale')
    settle = math.ceil(.5 / budget['step_duration_s'])
    required = gripper_steps + sum(math.ceil(len(segment['waypoints']) / speed) + settle
                                  for segment in segments)
    if budget['terminal'] or required > budget['remaining_steps']:
        raise MotionBudgetExceeded(budget, required)


def admit_place_motion(connector, plan, boundary, opening_steps):
    """Check the full route, but prioritize release over an optional retreat."""
    try:
        require_motion_budget(connector, plan.segments, gripper_steps=opening_steps)
        plan.full_route_fits_budget = True
    except MotionBudgetExceeded:
        require_motion_budget(connector, plan.segments[:boundary], gripper_steps=opening_steps)
        plan.full_route_fits_budget = False


def retreat_fits_budget(connector, segments):
    try:
        require_motion_budget(connector, segments)
        return True
    except MotionBudgetExceeded:
        return False
