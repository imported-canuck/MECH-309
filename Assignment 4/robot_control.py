"""
Open-loop vs. closed-loop (state-feedback) response for a discrete-time LTI system.

- Open-loop: no control (u_k = 0)  ->  x_{k+1} = A x_k
- Closed-loop: state feedback u_k = -K x_k  ->  x_{k+1} = (A - B K) x_k

Notation:
- Suffix _CL means "closed-loop" (with control).
- In the assignment, A_bar corresponds to A_CL.
"""

import numpy as np
from scipy import signal
from matplotlib import pyplot as plt


# Plot style (optional)
plt.rc("lines", linewidth=2)
plt.rc("axes", grid=True)
plt.rc("grid", linestyle="--")


def simulate_discrete_ss(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    D: np.ndarray,
    dt: float,
    t: np.ndarray,
    u: np.ndarray,
    x0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate discrete-time state-space system x_{k+1}=Ax_k+Bu_k, y_k=Cx_k+Du_k.

    Returns:
        xout: (N, n) state trajectory
        yout: (N, p) output trajectory
    """
    sys = signal.StateSpace(A, B, C, D, dt=dt)
    tout, yout, xout = signal.dlsim(sys, u, t=t, x0=x0)
    return xout, np.squeeze(yout)


def plot_state_response(t: np.ndarray, q: np.ndarray, v: np.ndarray, title_prefix: str) -> None:
    """Plot position and velocity vs time."""
    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(t, q)
    ax[1].plot(t, v, color="C1")

    ax[0].set_title(f"{title_prefix}: Position vs. Time")
    ax[1].set_title(f"{title_prefix}: Velocity vs. Time")

    ax[0].set_ylabel(r"$q(t_k)$ (m)")
    ax[1].set_ylabel(r"$v(t_k)$ (m/s)")
    ax[1].set_xlabel(r"$t$ (s)")

    fig.tight_layout()


def main():
    # Discrete-time system model
    dt = 0.1  # sample time
    A = np.array([[1.0, 0.09975042], [0.0, 0.99501248]], dtype=float)
    B = np.array([[0.00016639], [0.00332501]], dtype=float)
    C = np.eye(2)
    D = np.zeros((2, 1), dtype=float)

    # Simulation setup
    t = np.arange(0.0, 100.0, dt)
    N = t.size
    u = np.zeros(N)          # open-loop input: u_k = 0
    x0 = np.array([1.0, 0.1])  # initial condition: [position, velocity]

    # Open-loop (uncontrolled)
    Lambda = np.linalg.eigvals(A)
    print("Open-loop (uncontrolled) eigenvalues:", Lambda)
    print("All |lambda| < 1 ?", np.all(np.abs(Lambda) < 1))
    print("Spectral radius rho(A) =", np.max(np.abs(Lambda)), "\n")

    xout, yout = simulate_discrete_ss(A, B, C, D, dt, t, u, x0)
    q = xout[:, 0]
    v = xout[:, 1]
    plot_state_response(t, q, v, title_prefix="Open-loop (No Control)")

    # Closed-loop (controlled) via state feedback u = -Kx
    K = np.array([[29.77431187, 57.16016881]])  # Do not change this.

    # Closed-loop system matrix
    A_CL = A - B @ K

    Lambda_CL = np.linalg.eigvals(A_CL)
    print("Closed-loop (controlled) eigenvalues:", Lambda_CL)
    print("All |lambda| < 1 ?", np.all(np.abs(Lambda_CL) < 1))
    print("Spectral radius rho(A_CL) =", np.max(np.abs(Lambda_CL)), "\n")

    # Closed-loop simulation (we simulate the autonomous closed-loop dynamics x_{k+1}=A_CL x_k)
    xout_CL, yout_CL = simulate_discrete_ss(A_CL, B, C, D, dt, t, u, x0)
    q_CL = xout_CL[:, 0]
    v_CL = xout_CL[:, 1]
    plot_state_response(t, q_CL, v_CL, title_prefix="Closed-loop (With Control)")

    plt.show()

if __name__ == "__main__":
    main()