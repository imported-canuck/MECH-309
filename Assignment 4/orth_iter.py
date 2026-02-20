# %%
"""
Eigenvalue analysis for a matrix loaded from CSV.

What you will do:
(1) Load A from 'matrix.csv' and report its dimension.
(2) Compute eigenvalues/eigenvectors using SciPy (reference).
(3) Compute the condition number using SciPy (reference).
(4) Implement orthogonal iteration to approximate eigenpairs.
(5) (Optional) Estimate the condition number from your eigenvalues (if applicable).

IMPORTANT:
- Search for:  ### STUDENT TODO ###
  These are the lines you must modify.
"""
import numpy as np
from scipy import linalg
from typing import Tuple, Optional


# %%
def orthogonal_iteration(
    A: np.ndarray,
    Nmax: int = 500,
    tol: float = 1e-8,
    X0: Optional[np.ndarray] = None,
    store_history: bool = False,
) -> Tuple[np.ndarray, np.ndarray, int, Optional[np.ndarray]]:
    """
    Orthogonal iteration.
    """
    A = np.asarray(A)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square 2D array.")
    n = A.shape[0]

    # Initial orthonormal basis
    if X0 is None:
        X = np.eye(n)
    else:
        X0 = np.asarray(X0)
        ### STUDENT TODO ###
        X = X0

    X_hist = [X.copy()] if store_history else None
    ### STUDENT TODO ###
    # Implement Algorithm
    k = 0
    X_hist
    X_hist.append(X.copy())
    V = X0
    lam = np.einsum("ij,ij->j", V, A @ V)  # v_i^T A v_i (columnwise)

    return V, lam, k, (np.stack(X_hist, axis=0) if store_history else None)


# %%
# -----------------------------
# Load matrix from CSV
# -----------------------------
A = np.loadtxt("matrix.csv", dtype=float, delimiter=",")
n = A.shape[0]
print(f"Matrix dimension: n = {n}\n")

# ----------------------------
# (2) SciPy eigenvalues/eigenvectors (REFERENCE)
# -----------------------------
# ### STUDENT TODO ###
# Replace the placeholders

lam_scipy = np.zeros(n)          # <-- ### STUDENT TODO ###
V_scipy = np.zeros((n, n))       # <-- ### STUDENT TODO ###

print("Eigenvalues from SciPy (reference):\n", lam_scipy, end="\n\n")
print("Eigenvectors from SciPy (reference):\n", V_scipy, end="\n\n")

# -----------------------------
# (3) Condition number (REFERENCE)
# -----------------------------
# ### STUDENT TODO ###
# Replace placeholder

kappa_scipy = 1.0                # <-- ### STUDENT TODO ###
print("Condition number of A (SciPy reference):", kappa_scipy, end="\n\n")


# -----------------------------
# (4) Orthogonal iteration (YOUR IMPLEMENTATION)
# -----------------------------
# Here, you run the provided orthogonal iteration function.
# You should choose tol and Nmax sensibly.

Nmax = 500
tol = 1e-5
V, lam, iters, _ = orthogonal_iteration(A, Nmax=Nmax, tol=tol, store_history=False)

print("Orthogonal iteration iterations:", iters)
print("Approximate eigenvalues (orthogonal iteration):\n", lam, end="\n\n")
print("Approximate eigenvectors (orthogonal iteration):\n", V)


# %%
# -----------------------------
# (5) Condition number estimate from eigenvalues (OPTIONAL / DEPENDS ON A)
# -----------------------------
# ### STUDENT TODO ###
# Only do this if it makes sense for your matrix type.
# Example: if A is symmetric positive definite (SPD), then:
#     kappa_2(A) = lambda_max(A) / lambda_min(A)
# Otherwise, you may need singular values instead.

kappa_est = 1.0                  # <-- ### STUDENT TODO ###
print("Estimated condition number of A (from your eigenvalues):", kappa_est, end="\n\n")