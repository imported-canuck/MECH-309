import numpy as np

# Time integration
def heun_step(state: np.ndarray, dt: float, rhs_func) -> np.ndarray:
    """
    Advance one time step with Heun's method.
    """
    k1 = rhs_func(state)
    k2 = rhs_func(state + dt * k1)
    return state + 0.5 * dt * (k1 + k2)

def solve_heun(
    initial_state: np.ndarray,
    t_start: float,
    t_end: float,
    dt: float,
    rhs_func,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve an ODE system with Heun's method.
    """
    time = np.arange(t_start, t_end + dt, dt, dtype=float)
    n_steps = time.size
    n_state = initial_state.shape[0]

    states = np.zeros((n_state, n_steps), dtype=float)
    states[:, [0]] = np.asarray(initial_state, dtype=float).reshape(n_state, 1)

    for k in range(n_steps - 1):
        states[:, [k + 1]] = heun_step(states[:, [k]], dt, rhs_func)

    return time, states