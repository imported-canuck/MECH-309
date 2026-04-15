import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

import heun

# Plotting
def plot_orbit(positions: np.ndarray, eccentricity: float) -> None:
    """
    Plot the orbit in the x-y plane.
    """
    fig, ax = plt.subplots()
    ax.set_xlabel(r"$x$ (normalized position)")
    ax.set_ylabel(r"$y$ (normalized position)")
    ax.set_title(f"Orbit for e = {eccentricity:1.1f}")

    ax.plot(0.0, 0.0, "o", label="Earth Centre", color="C2")
    ax.plot(positions[0, :], positions[1, :], "-", label="orbit", color="C0")
    ax.plot(positions[0, 0], positions[1, 0], "d", label="start", color="C1")
    ax.plot(positions[0, -1], positions[1, -1], "d", label="end", color="C3")

    ax.axis("equal")
    ax.legend(loc="lower left")
    fig.tight_layout()
    plt.show()


def plot_time_series(
    time: np.ndarray,
    values: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    label: str,
) -> None:
    """
    Plot a generic time series.
    """
    fig, ax = plt.subplots()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.plot(time, values, label=label)
    ax.legend(loc="lower left")
    fig.tight_layout()
    plt.show()

# orbital
def orbital_rhs(state: np.ndarray, gm: float = 1.0) -> np.ndarray:
    """
    Right-hand side of the 2D orbital equations of motion.
    """
    state = np.asarray(state, dtype=float).reshape(4, 1)

    position = state[0:2, :]
    velocity = state[2:4, :]

    radius = np.linalg.norm(position)
    acceleration = -gm * position / radius**3

    return np.vstack((velocity, acceleration))


def orbital_energy(state: np.ndarray, gm: float = 1.0) -> float:
    """
    Compute the total mechanical energy of the orbit.
    """
    state = np.asarray(state, dtype=float).reshape(4, 1)

    position = state[0:2, :]
    velocity = state[2:4, :]

    radius = np.linalg.norm(position)
    kinetic = 0.5 * (velocity.T @ velocity).item()
    potential = -gm / radius

    return kinetic + potential


def initial_state_from_eccentricity(eccentricity: float) -> np.ndarray:
    """
    Construct the initial state for a normalized Kepler orbit.
    """
    e = eccentricity
    if not (0.0 <= e < 1.0):
        raise ValueError("Eccentricity must satisfy 0 <= e < 1.")

    return np.array(
        [[1.0 - e],
         [0.0],
         [0.0],
         [np.sqrt((1.0 + e) / (1.0 - e))]],
        dtype=float,
    )

# Diagnostics
def compute_state_error(states: np.ndarray, ord_norm: int = 1) -> float:
    """
    Compute the difference between the initial and final full state.
    """
    return float(np.linalg.norm(states[:, -1] - states[:, 0], ord=ord_norm))


def solve_heun_exact_end(
    initial_state: np.ndarray,
    t_start: float,
    t_end: float,
    dt: float,
    rhs_func,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Heun solve, but force the last time point to land exactly at t_end.
    This is useful here because 2*pi is not an integer multiple of dt=0.001.
    """
    times = [t_start]
    states = [np.asarray(initial_state, dtype=float).reshape(4, 1)]

    t = t_start
    while t < t_end - 1e-15:
        h = min(dt, t_end - t)
        next_state = heun.heun_step(states[-1], h, rhs_func)
        states.append(next_state)
        t += h
        times.append(t)

    return np.array(times), np.hstack(states)

def compute_position_error(states: np.ndarray, ord_norm: int = 1) -> float:
    """
    Compute the difference between the initial and final position only.
    """
    return float(np.linalg.norm(states[0:2, -1] - states[0:2, 0], ord=ord_norm))


def compute_energy_history(states: np.ndarray, gm: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the energy history and relative energy error.
    """
    n_steps = states.shape[1]
    energies = np.zeros(n_steps, dtype=float)

    for k in range(n_steps):
        energies[k] = orbital_energy(states[:, k], gm=gm)

    initial_energy = energies[0]
    relative_error = np.abs(energies - initial_energy) / np.abs(initial_energy)

    return energies, relative_error


if __name__ == "__main__":
    # Problem setup
    eccentricity = 0.9
    t_start = 0.0
    t_end = 2.0 * np.pi
    dt = 0.001

    initial_state = initial_state_from_eccentricity(eccentricity)
    # Solve
    time, states = solve_heun_exact_end(
        initial_state=initial_state,
        t_start=t_start,
        t_end=t_end,
        dt=dt,
        rhs_func=orbital_rhs,
    )

    # Post-processing
    positions = states[0:2, :]
    energies, energy_rel_error = compute_energy_history(states)
    position_error = compute_position_error(states, ord_norm=1)

    print(
        f"The position error when e = {eccentricity} after one orbit is "
        f"{position_error:.6e} (units)"
    )

    # Plotting
    plot_orbit(positions, eccentricity)

    plot_time_series(
        time=time,
        values=energies,
        xlabel=r"$t$ (normalized)",
        ylabel="Energy (normalized)",
        title=f"Energy vs time for e = {eccentricity:1.1f}",
        label="Energy",
    )

    plot_time_series(
        time=time,
        values=energy_rel_error,
        xlabel=r"$t$ (normalized)",
        ylabel="Relative energy error",
        title=f"Energy Relative Error vs time for e = {eccentricity:1.1f}",
        label="Energy Relative Error",
    )