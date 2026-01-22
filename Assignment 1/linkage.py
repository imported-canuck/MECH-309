from typing import Union, Literal
import numpy as np
import numpy.typing as npt

ArrayLike = Union[float, npt.NDArray[np.floating]]

def linkage(
    psi: ArrayLike,
    a: float,
    b: float,
    c: float,
    d: float,
    branch: Literal[-1, 1] = -1,
    eps: float = 1e-12
) -> npt.NDArray[np.floating]:
    """
    Four-bar linkage position analysis.

    Parameters
    ----------
    psi : float or ndarray
        Input angle(s) in radians.
    a, b, c, d : float
        Link lengths (must be > 0).
    branch : {-1, 1}, optional
        Assembly mode selector.
    eps : float, optional
        Tolerance for discriminant clipping.

    Returns
    -------
    ndarray
        Array of shape (2, ...) containing [phi, beta].
    """
    psi = np.asarray(psi, dtype=float)

    if not (a > 0 and b > 0 and c > 0 and d > 0):
        raise ValueError("Link lengths a, b, c, d must be positive.")
    if branch not in (-1, 1):
        raise ValueError("branch must be +1 or -1.")

    # constants
    k1 = d / a
    k2 = d / c
    k3 = (a*a - b*b + c*c + d*d) / (2*a*c)
    k4 = d / b
    k5 = (c*c - d*d - a*a - b*b) / (2*a*b)

    Q = np.cos(psi)
    S = np.sin(psi)

    # coefficients
    Aphi = k3 - k1 - (k2 - 1) * Q
    Bphi = -2 * S
    Cphi = k3 + k1 - (k2 + 1) * Q

    Abeta = k5 - k1 + (k4 + 1) * Q
    Bbeta = Bphi
    Cbeta = k5 + k1 + (k4 - 1) * Q

    disc_phi = Bphi**2 - 4*Aphi*Cphi
    disc_beta = Bbeta**2 - 4*Abeta*Cbeta

    if np.any(disc_phi < -eps) or np.any(disc_beta < -eps):
        raise ValueError("No real solution for given psi and link lengths.")

    sqrt_phi = np.sqrt(np.maximum(disc_phi, 0.0))
    sqrt_beta = np.sqrt(np.maximum(disc_beta, 0.0))

    phi = 2 * np.arctan2(-Bphi + branch * sqrt_phi, 2 * Aphi)
    beta = 2 * np.arctan2(-Bbeta + branch * sqrt_beta, 2 * Abeta)

    return np.stack((phi, beta), axis=0)