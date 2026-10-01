"""Physics time for capture epochs and recording, independent of control counters."""


def simulation_time_s(env):
    clock = getattr(env, "get_simulation_time_s", None)
    if callable(clock):
        return float(clock())
    # Fall back to the MuJoCo clock exposed by compatible environments and adapters.
    return float(env.handle.env.sim.data.time)
