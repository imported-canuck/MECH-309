# Packages
import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg

def f_temp(T0, TE, k, t):
    # Newton's law of cooling
    return TE + (T0 - TE) * np.exp(-k * t)


def f(T0, TE, k, t, T_des):
    # Function for root finding
    # TODO: return the function whose root you want to compute
    return f_temp(T0, TE, k, t) - T_des


def df_dt(T0, TE, k, t):
    # TODO: derivative needed if Newton's method is used
    return -(T0 - TE) * k * np.exp(-k * t)


# %%
# Only run the solver and plotting when executed directly
if __name__ == "__main__":

    # Form the root finding problem
    k = 1        # TODO: replace with value from nonlinear least squares
    T0 = 98      # TODO: replace with value from nonlinear least squares
    TE = 22      # Temp of environment
    T_des = 35   # Desired temperature

    eps1 = 1e-6  # Convergence tolerance. Don't change
    Nmax = 50    # Max iterations. Don't change

    tIC = 5.5            # initial guess if using Newton / secant
    aIC, bIC = 10, 20    # initial bounds if using bisection

    t = [tIC]
    i = 0  # counter

    # TODO: Write your root finding code here
    while i < Nmax:

        ft = f(T0, TE, k, t[-1], T_des)

        # Example Newton step
        dft = df_dt(T0, TE, k, t[-1])

        t_new = t[-1] - ft / dft

        t.append(t_new)

        if abs(t[-1] - t[-2]) < eps1:
            break

        i += 1

    print("Converged value is", t[-1])
    print("Number of iterations is", i)

    # Optional plotting

    t_plot = np.linspace(0, 30, 200)
    T_plot = f_temp(T0, TE, k, t_plot)

    fig, ax = plt.subplots()
    ax.plot(t_plot, T_plot, label="Temperature")
    ax.axhline(T_des, linestyle="--", label="Desired temperature")
    ax.axvline(t[-1], linestyle="--", label="Root")
    ax.set_xlabel("t")
    ax.set_ylabel("T(t)")
    ax.legend()
    fig.tight_layout()

    plt.show()