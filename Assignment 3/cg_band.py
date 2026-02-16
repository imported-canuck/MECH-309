import numpy as np

def band_matvec_sym(a, b, c, x):
    """
    y = A x for symmetric pentadiagonal A.
    """
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    n = x.shape[0]
    y = np.zeros_like(x)

    # TODO: STUDENT implement a banded A x

    return y


def cg_solve_banded(a, b, c, rhs, x0, *, tol=1e-8, max_iter=200, M_inv=None):
    """
    Preconditioned CG for symmetric positive definite pentadiagonal matrices.

    Solves Ax = rhs, where A is symmetric pentadiagonal given by diagonals (a,b,c).
    Optional preconditioner: M_inv(r) ≈ A^{-1} r (apply inverse of preconditioner).
    """
    rhs = np.asarray(rhs, dtype=float).reshape(-1, 1)
    x = np.asarray(x0, dtype=float).reshape(-1, 1)

    def Ax(v):
        return band_matvec_sym(a, b, c, v)

    # Initial residual
    r = rhs - Ax(x)

    # Apply preconditioner (or identity)
    if M_inv is None:
        z = r.copy()
    else:
        z = np.asarray(M_inv(r), dtype=float).reshape(-1, 1)

    p = z.copy()
    rz = (r.T @ z).item()

    for iters in range(max_iter):
        # TODO: STUDENT: complete the CG iteration

    return x.ravel(), iters

if __name__ == "__main__":
    import read_matrix as rm
    A = rm.load_matrix("ToStudents/matrix_b.csv")
    b = A[:, -1].reshape(-1, 1)
    A = A[:, :-1]