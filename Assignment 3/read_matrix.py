import numpy as np

def load_matrix(filename):
    """
    Load a matrix from a CSV file and compute its 1-norm condition number.

    Parameters
    ----------
    filename : str
        Path to the CSV file containing the matrix.
    """
    # Load matrix
    A = np.loadtxt(filename, dtype=float, delimiter=',')

    # Matrix dimension
    m, n = A.shape
    if m == n:
        print('The dimension n of the square A matrix is', n)
    else:
        print('The dimension n of the square A matrix is', m, 'and there is also a b vector of size',m,'x',n-m)
        print('Split using: A_mat = A[:, :-1]; b_vec = A[:, -1].reshape(-1, 1)')
    return A