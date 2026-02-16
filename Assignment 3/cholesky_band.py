import numpy as np

def banded_cholesky_factor(a, b, c):
    """
    Cholesky factorization A = L L^T for symmetric SPD pentadiagonal A, with pentadiagonal.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)

    n = c.size
    if not (b.size == n - 1 and a.size == n - 2):
        raise ValueError("Lengths must be: c=n, b=n-1, a=n-2")

    p = np.zeros(n)
    l = np.zeros(n - 1)
    m = np.zeros(n - 2)

    # TODO: STUDENT implement your sparse cholesky factorization

    return m, l, p


def banded_cholesky_solve(a, b, c, rhs):
    """
    Solve Ax = rhs for symmetric SPD pentadiagonal A using compact Cholesky.
    rhs can be (n,) or (n,k).
    """
    m, l, p = banded_cholesky_factor(a, b, c)

    rhs = np.asarray(rhs, dtype=float)
    if rhs.ndim == 1:
        rhs = rhs.reshape(-1, 1)
    n, k = rhs.shape
    if n != p.size:
        raise ValueError("rhs length must match matrix dimension")

    # Forward: Ly = rhs
    y = np.zeros((n, k))
    # TODO: STUDENT implement your forward solve

    # Backward: L^T x = y
    x = np.zeros((n, k))
    # TODO: STUDENT implement your backward solve

    return x.ravel() if k == 1 else x

if __name__ == "__main__":
    import read_matrix as rm
    A = rm.load_matrix("ToStudents/matrix_b.csv")
    b = A[:, -1].reshape(-1, 1)
    A = A[:, :-1]