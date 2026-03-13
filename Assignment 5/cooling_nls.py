import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg

def f_temp(T0, TE, k, t):
    # Newton's law of cooling
    return TE + (T0 - TE) * np.exp(-k * t)


def Jacobian(T0, TE, k, t):
    # Jacobian wrt T0 and k associated with Newton's law of cooling
    J1 = 1  # TODO: derivative wrt T0
    J2 = 1  # TODO: derivative wrt k
    return np.array([J1, J2])

if __name__ == "__main__":
    # Load and extract data
    data = np.loadtxt(
        'temp_vs_time.csv',
        dtype=float,
        delimiter=',',
        skiprows=1,
        usecols=(0, 1,)
    )

    t = data[:, 0]  # seconds
    T = data[:, 1]  # temperature

    fig, ax = plt.subplots()
    ax.set_title(r'Temperature vs. Time')
    ax.set_xlabel(r'$t$ (min)')
    ax.set_ylabel(r'$T(t)$ ($^\circ$C)')
    ax.plot(t, T, label='temperature', color='C0')
    ax.plot(t, T, 'x', color='C3')
    ax.legend(loc='upper right')
    fig.tight_layout()
    plt.show()

    # Form the nonlinear least squares problem

    m = T.shape[0]
    n = 2

    TE = 22  # environment temperature
    eps1 = 1e-6
    Nmax = 10

    b = np.ones((m, 1))  # TODO: fill with measurement vector
    A = np.zeros((m, n))  # Jacobian matrix

    j = 0

    b_hat = np.zeros((m, 1))  # model prediction
    x_hat = np.array([[98],
                      [0.05]])  # initial guess for T0 and k

    delta_x = np.array([[1],
                        [0.0]])  # initial delta_x

    # Nonlinear least squares loop
    while (linalg.norm(delta_x, 2) > eps1) and (j < Nmax):

        T0_hat = x_hat[0, 0]
        k_hat = x_hat[1, 0]

        # TODO: compute model prediction
        b_hat = np.random.rand(m, 1)

        # TODO: build Jacobian matrix
        A = np.random.rand(m, n)

        # TODO: solve linearized least squares problem
        x_hat = np.array([[98],
                          [0.05]])  # dummy update

        j += 1

    print("Converged value is", x_hat.T)
    print("Number of iterations is", j)