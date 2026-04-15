import os
import subprocess
import shutil
import numpy as np
from scipy import linalg
from matplotlib import pyplot as plt

# Visualisation Helpers
def save_snapshot(
    u: np.ndarray,
    t: float,
    x: np.ndarray,
    y: np.ndarray,
    step: int,
    output_dir: str = "figs",
    vmin: float = 0,
    vmax: float = 10
) -> None:
    """
    Plot one snapshot of u(x,y,t).
    """
    plt.figure(figsize=(6, 5))
    plt.imshow(
        u.T,
        origin="lower",
        extent=[x[0], x[-1], y[0], y[-1]],
        aspect="auto",
        vmin=vmin,
        vmax=vmax
    )
    plt.colorbar(label="u(x,y,t)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Brusselator: u at t = {t:.2f}")
    plt.tight_layout()
    filename = os.path.join(output_dir, f"snapshot_{step:05d}.png")
    plt.savefig(filename, dpi=150)
    #plt.show()
    plt.close()


def create_video_ffmpeg(
    input_pattern="figs/snapshot_*.png",
    output_file="movie.mp4",
    fps=10
):
    # Check if ffmpeg exists (on both windows and mac you might need to install it)
    if shutil.which("ffmpeg") is None:
        print("ffmpeg not found. Skipping video creation.")
        return
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate", str(fps),
        "-pattern_type","glob",
        "-i", input_pattern,
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "slow",
        "-pix_fmt", "yuv420p",
        output_file
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"Video saved as {output_file}")
    except subprocess.CalledProcessError:
        print("ffmpeg failed to create video.")

#########################
# Brusselator
#########################


def laplacian_periodic(phi: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    2D Laplacian using second-order central finite differences
    with periodic boundary conditions.
    """
    # Central difference in the x-direction plus central difference in the y-direction.
    # np.roll(...) implements periodic wrap-around automatically.
    return (
        (np.roll(phi, -1, axis=0) - 2.0 * phi + np.roll(phi, 1, axis=0)) / dx**2
        + (np.roll(phi, -1, axis=1) - 2.0 * phi + np.roll(phi, 1, axis=1)) / dy**2
    )

def brusselator_rhs(
    u: np.ndarray,
    v: np.ndarray,
    A: float,
    B: float,
    Du: float,
    Dv: float,
    dx: float,
    dy: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Right-hand side of the semi-discrete Brusselator system.
    """
    # Compute the discrete Laplacians of u and v.
    lap_u = laplacian_periodic(u, dx, dy)
    lap_v = laplacian_periodic(v, dx, dy)

    # Reaction terms from the Brusselator model.
    reaction_u = A - (B + 1.0) * u + u**2 * v
    reaction_v = B * u - u**2 * v

    # Add diffusion terms.
    du_dt = reaction_u + Du * lap_u
    dv_dt = reaction_v + Dv * lap_v

    return du_dt, dv_dt

def rk_step(
    u: np.ndarray,
    v: np.ndarray,
    dt: float,
    A: float,
    B: float,
    Du: float,
    Dv: float,
    dx: float,
    dy: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    One classical fourth-order Runge-Kutta step for the semi-discrete PDE system.
    """
    # Stage 1
    k1u, k1v = brusselator_rhs(u, v, A, B, Du, Dv, dx, dy)

    # Stage 2
    k2u, k2v = brusselator_rhs(
        u + 0.5 * dt * k1u,
        v + 0.5 * dt * k1v,
        A, B, Du, Dv, dx, dy
    )

    # Stage 3
    k3u, k3v = brusselator_rhs(
        u + 0.5 * dt * k2u,
        v + 0.5 * dt * k2v,
        A, B, Du, Dv, dx, dy
    )

    # Stage 4
    k4u, k4v = brusselator_rhs(
        u + dt * k3u,
        v + dt * k3v,
        A, B, Du, Dv, dx, dy
    )

    # Combine the four RK4 stages.
    u_new = u + (dt / 6.0) * (k1u + 2.0 * k2u + 2.0 * k3u + k4u)
    v_new = v + (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)

    return u_new, v_new

def initialize_fields(
    Nx: int,
    Ny: int,
    A: float,
    B: float,
    noise_amplitude: float,
    seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """
    Initial condition:
        u(x,y,0) = 1 + 0.05 * eta_1(x,y)
        v(x,y,0) = 3.4 + 0.05 * eta_2(x,y)
    with eta_i in [-1, 1].
    """
    rng = np.random.default_rng(seed)
    eta1 = rng.uniform(-1.0, 1.0, size=(Nx, Ny))
    eta2 = rng.uniform(-1.0, 1.0, size=(Nx, Ny))

    u0 = A + noise_amplitude * eta1
    v0 = B / A + noise_amplitude * eta2
    return u0, v0

def run_brusselator_2d(
    Nx: int = 100,
    Ny: int = 100,
    A: float = 2.5,
    B: float = 10.0,
    Du: float = 0.004,
    Dv: float = 0.002,
    Lx: float = 5.0,
    Ly: float = 5.0,
    t0: float = 0.0,
    t_end: float = 40.0,
    dt: float = 5e-3,
    noise_amplitude: float = 0.005,
    seed: int = 42,
    snapshot_times=None
):
    """
    Run the 2D Brusselator and return snapshots for notebook visualization.
    """
    if snapshot_times is None:
        snapshot_times = [0.0, 5.0, 10.0, 20.0, 30.0, 40.0]

    # Grid spacing.
    dx = Lx / Nx
    dy = Ly / Ny

    # Grid coordinates.
    x = np.linspace(0.0, Lx, Nx, endpoint=False)
    y = np.linspace(0.0, Ly, Ny, endpoint=False)

    # Number of time steps.
    n_steps = int((t_end - t0) / dt)

    # Initial condition near the homogeneous steady state.
    u, v = initialize_fields(
        Nx, Ny, A, B,
        noise_amplitude=noise_amplitude,
        seed=seed
    )

    # Store snapshots by the corresponding time.
    snapshot_steps = {int(round((time - t0) / dt)): time for time in snapshot_times}
    u_snaps = {}
    v_snaps = {}

    # Save the initial condition if requested.
    if 0 in snapshot_steps:
        u_snaps[snapshot_steps[0]] = u.copy()
        v_snaps[snapshot_steps[0]] = v.copy()

    # Time integration loop.
    for n in range(1, n_steps + 1):
        u, v = rk_step(u, v, dt, A, B, Du, Dv, dx, dy)

        if n in snapshot_steps:
            t = snapshot_steps[n]
            u_snaps[t] = u.copy()
            v_snaps[t] = v.copy()

    return x, y, u_snaps, v_snaps, u, v

if __name__ == "__main__":
    # Parameters
    A, B = 2.5,10.0
    Du, Dv = 0.004, 0.002

    # Domain and grid
    Lx, Ly = 5.0, 5.0
    Nx, Ny = 101, 101 # TODO: use smaller grid during development
    dx, dy = Lx/Nx, Ly/Ny

    x = np.linspace(0.0, Lx, Nx, endpoint=False)
    y = np.linspace(0.0, Ly, Ny, endpoint=False)

    # Time integration
    t0 = 0.0
    t_end = 40.0 # TODO: use smaller time during development
    dt = 5e-3
    n_steps = int((t_end - t0) / dt)

    # Initial condition
    u, v = initialize_fields(Nx, Ny, A, B, noise_amplitude=0.005, seed=42)

    # Times at which snapshots should be shown
    snapshot_times = np.linspace(0.0, 40, 81) # This is nice for generating a video

    snapshot_steps = {int(time / dt): time for time in snapshot_times}
    # Initial snapshot
    save_snapshot(u, t0, x, y, step=0)

    # Time loop
    for n in range(1, n_steps + 1):
        u, v = rk_step(u, v, dt, A, B, Du, Dv, dx, dy)
        t = n * dt

        if n in snapshot_steps:
            print(f"Saving snapshot at t = {t:.3f}")
            save_snapshot(u, t, x, y, step=n)
            print(f"u_min = {u.min():.6f}, u_max = {u.max():.6f}, u_mean = {u.mean():.6f}")
            print(f"v_min = {v.min():.6f}, v_max = {v.max():.6f}, v_mean = {v.mean():.6f}")

    create_video_ffmpeg()