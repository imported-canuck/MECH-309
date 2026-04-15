import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root

# Numerical derivatives
def dD_dl(l, d, h=1e-6):
    return (drag(l + h, d) - drag(l - h, d)) / (2.0 * h)


def dD_dd(l, d, h=1e-6):
    return (drag(l, d + h) - drag(l, d - h)) / (2.0 * h)

# functions:
def drag(l, d):
    if l <= 0 or d <= 0 or d >= l:
        return np.inf

    e2 = 1.0 - (d / l)**2
    if e2 <= 0:
        return np.inf

    e = np.sqrt(e2)

    return (
        1.0
        + 1.5 * (d / l)**(3.0 / 2.0)
        + 7.0 * (d / l)**3
    ) * (
        d**2 + (d * l / e) * np.arcsin(e)
    )

def volume(l, d):
    return (np.pi / 6.0) * d**2 * l

def constraint(l, d):
    return volume(l, d) - 2.5

# KKT system
def kkt_system(x):
    l, d, lam = x

    # Avoid invalid region
    if l <= 0 or d <= 0 or d >= l:
        return np.array([1e6, 1e6, 1e6])

    # Using dD_dl, dD_dd and contrain build the kkt system
    eq1 = dD_dl(l, d) + lam * (np.pi / 6.0) * d**2
    eq2 = dD_dd(l, d) + lam * (np.pi / 3.0) * d * l
    eq3 = constraint(l, d)

    return np.array([eq1, eq2, eq3])


def main():
    # Initial guess
    x0 = np.array([3.0, 1.0, -100.0])

    sol = root(kkt_system, x0)

    if not sol.success:
        raise RuntimeError(f"Nonlinear solve failed: {sol.message}")

    l_opt, d_opt, lam_opt = sol.x

    print("Optimal solution:")
    print(f"  l*      = {l_opt:.6f} m")
    print(f"  d*      = {d_opt:.6f} m")
    print(f"  lambda* = {lam_opt:.6f}")
    print(f"  D(l*,d*) = {drag(l_opt, d_opt):.6f} N")
    print(f"  V(l*,d*) = {volume(l_opt, d_opt):.6f} m^3")


# Plotting
def plot_optimization_contours(L, Dd, Z, G, l_opt, d_opt,
                               n_levels=20,
                               percentile=85):
    """
    Plot objective contours and constraint curve with optimum.
    """

    plt.figure(figsize=(8, 6))

    # Objective contours
    z_min = np.nanmin(Z)
    z_max = np.nanpercentile(Z, percentile)
    levels = np.linspace(z_min, z_max, n_levels)

    cs = plt.contour(L, Dd, Z, levels=levels)
    plt.clabel(cs, inline=True, fontsize=8)

    # Constraint curve g(l,d)=0
    cc = plt.contour(L, Dd, G, levels=[0.0], linewidths=2)
    plt.clabel(cc, fmt={0.0: r"$g(\ell,d)=0$"})

    # Mark optimum
    plt.plot(l_opt, d_opt, 'ro', markersize=8, label='Optimum')

    plt.xlabel(r'$\ell$ [m]')
    plt.ylabel(r'$d$ [m]')
    plt.title('Objective contours and active constraint')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Call the optimizer main function
    l_opt, d_opt, lam_opt = main()

    # Create grid
    l_vals = np.linspace(1, 6, 200)
    d_vals = np.linspace(0.5, 3, 200)
    L, Dd = np.meshgrid(l_vals, d_vals)

    # Example objective (toy function)
    Z = np.vectorize(drag)(L, Dd).astype(float)
    Z[~np.isfinite(Z)] = np.nan   # convert inf to nan

    # Constraint (e.g. volume-like)
    G = constraint(L, Dd)

    # Call function
    plot_optimization_contours(L, Dd, Z, G, l_opt, d_opt)