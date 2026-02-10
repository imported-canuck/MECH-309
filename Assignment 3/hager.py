
#%% 
# Norms
import numpy as np

# matrix 1-norm
def matrix_1_norm(A):
    """
    Matrix 1-norm = maximum absolute column sum
    """
    # TODO: STUDENT: temporary, replace this.
    m1A = np.linalg.norm(A, 1)
    return m1A


# vector 1-norm
def vector_1_norm(x):
    """
    Vector 1-norm = sum of absolute values
    """
    # TODO: STUDENT: temporary, replace this.
    v1x = np.linalg.norm(x, 1) 
    return v1x


# vector infinity norm
def vector_inf_norm(x):
    """
    Vector infinity norm = maximum absolute value
    """
    # TODO: STUDENT: temporary, replace this.
    vinfx = np.linalg.norm(x, np.inf)
    return vinfx

#%% 
# Solver
import numpy as np

def LU_solve(A, b):
    """
    Solve Ax = b using LU decomposition with pivoting.
    """
    # TODO: STUDENT: temporary implementation, replace with your own LU solver.
    # Do NOT use numpy.linalg.solve in your final solution.
    x = np.linalg.solve(A, b)
    return x


def chol_solve(A, b):
    """
    Solve Ax = b using Cholesky decomposition.
    """
    # TODO: STUDENT: temporary implementation, replace with your own Cholesky solver.
    # Do NOT use numpy.linalg.solve in your final solution.
    x = np.linalg.solve(A, b)
    return x

#%%
# Hager Algorithm

def hager_matrix_1_norm(A, eps=1e-5, N_max=100):
    """
    Compute the matrix 1-norm ||A||_1 using Hager's algorithm.
    """
    n = A.shape[0]

    # initial vector
    x = np.ones((n, 1)) / n
    error = np.inf
    i = 0

    while (error >= eps) and (i <= N_max):
        # TODO: STUDENT
        # Implement Hager's algorithm iteration here
        i += 1

    # TODO: STUDENT: temporary, replace this
    norm_1_A_compute = 0

    return norm_1_A_compute



from norm import matrix_1_norm

def hager_inverse_1_norm(A, eps=1e-5, N_max=100):
    """
    Compute ||A^{-1}||_1 using the modified Hager algorithm
    and return an approximation of the condition number.
    """
    n = A.shape[0]

    # initial vector
    x = np.ones((n, 1)) / n
    error = np.inf
    i = 0

    while (error >= eps) and (i <= N_max):
        # TODO: STUDENT
        # Implement modified Hager algorithm iteration here
        i += 1

    # TODO: STUDENT: temporary, replace this
    norm_1_B_compute = 0  # B = A^{-1}

    kappa_app = matrix_1_norm(A) * norm_1_B_compute

    return norm_1_B_compute, kappa_app


if __name__ == "__main__":
    import read_matrix as rm
    A = rm.load_matrix("matrix_a.csv")