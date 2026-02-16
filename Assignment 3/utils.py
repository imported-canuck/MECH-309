import numpy as np

def forward_sub(A, b):
    """
    Solve the equation Ax = b for x, where A is a lower triangular matrix. 
    Assumes a lower triangular matrix A. Expects a that A is non-singular, 
    and that the dimensions of A and b are compatible (but checks these and 
    throws an exception if not).
    
    Parameters:
    A (ndarray): A lower triangular matrix of shape (n, n).
    b (ndarray): A vector of shape (n, 1).
    
    Returns:
    x (ndarray): The solution vector of shape (n, 1).
    """
    tol = 1e-12                  # Tolerance to avoid floating point issues
    n = A.shape[0]           # Size of matrix A

    if A.shape[1] != n:      # Check if A is square
        raise ValueError("Matrix A must be square.")
    if b.shape[0] != n:      # Check if dimensions of b are compatible
        raise ValueError("Vector b must have compatible dimensions with matrix A.")

    x = np.zeros((n, 1))     # Initialize solution vector x as a column vector
    
    for i in range(n):       # Loop over each row
        if np.abs(A[i, i]) <= tol:     # Check for singularity (zero diagonal element)
            raise ValueError("Matrix is singular.")
        
        # For each column of row i before the diagonal subtract the product of 
        # the entry A(i, j) and the corresponding entry of vector x from b[i]
        for j in range(i):
            b[i, 0] -= A[i, j] * x[j, 0] 

        # Finaly divide b[i] by the diagonal element A(i, i) to get x[i]
        x[i, 0] = b[i, 0] / A[i, i]  

    return x 

def backward_sub(A, b):
    """
    Solve the equation Ax = b for x, where A is an upper triangular matrix. 
    Assumes an upper triangular matrix A. Expects a that A is non-singular, 
    and that the dimensions of A and b are compatible (but checks these and 
    throws an exception if not).
    
    Parameters:
    A (ndarray): An upper triangular matrix of shape (n, n).
    b (ndarray): A vector of shape (n, 1).
    
    Returns:
    x (ndarray): The solution vector of shape (n, 1).
    """
    tol = 1e-12                  # Tolerance to avoid floating point issues    
    n = A.shape[0]  
                 # Size of matrix A
    if A.shape[1] != n:          # Check if A is square
        raise ValueError("Matrix A must be square.")
    if b.shape[0] != n:          # Check if dimensions of b are compatible
        raise ValueError("Vector b must have compatible dimensions with matrix A.")

    x = np.zeros((n, 1))               # Initialize solution vector x
    
    for i in range(n - 1, -1 , -1):    # Loop over each row from bottom to top
        # Check for singularity (zero diagonal element)
        if np.abs(A[i, i]) <= tol:         
            raise ValueError("Matrix is singular.")

        # For each column of row i after the diagonal subtract the product of 
        # the entry A(i, j) and the corresponding entry of vector x from b[i]   
        for j in range(i + 1, n):
            b[i, 0] -= A[i, j] * x[j, 0]
        # Finaly divide b[i] by the diagonal element A(i, i) to get x[i]
        x[i, 0] = b[i, 0] / A[i, i]
           
    return x

def gaussian_elimination(A, b):
    """
    Solve the equation Ax = b for x using Gaussian elimination with partial 
    pivoting. Expects a that A is non-singular, and that the dimensions of A 
    and b are compatible (but checks these and throws an exception if not).
    
    Parameters:
    A (ndarray): An matrix of shape (n, n).
    b (ndarray): A vector of shape (n, 1).
    
    Returns:
    x (ndarray): The solution vector of shape (n, 1).
    """    
    tol = 1e-12                  # Tolerance to avoid floating point issues
    n = A.shape[0]               # Size of matrix A
    
    if A.shape[1] != n:          # Check if A is square
        raise ValueError("Matrix A must be square.")
    if b.shape[0] != n:          # Check if dimensions of b are compatible
        raise ValueError("Vector b must have compatible dimensions with matrix A.")

    for i in range(n - 1):       # Loop over all rows of A (apart from last, 
                                 # since A is already upper-triangular by then)

    # Start partial pivoting: find the row with the largest element 
    # at the column position of the pivot (don't want small numbers on pivot)
        p = i + np.argmax(np.abs(A[i:, i]))  
    
    # If no row with a nonzero entry on the pivot exists, matrix is singular    
        if np.abs(A[p, i]) <= tol:           
                raise ValueError("Matrix is singular.")
        
    # If the current row is not the one with the greatest pivot element, 
    # do partial pivoting by swapping current "row i" with row with greatest 
    # pivot element "row p". And apply the same operation on vector b       
        if p != i:               
                A[[i, p], :] = A[[p, i], :]  # Row swap "i" and "p" on A
                b[[i, p], :] = b[[p, i], :]  # Swap corresponding entries in b

    # End partial pivoting, now eliminate entries below the pivot:
    # For each subsequent row (starting at i + 1) compute the factor that 
    # would eliminate the element of row j that is below the pivot
        for j in range(i + 1, n):       
            factor = A[j, i] / A[i, i]   # Compute elimination factor
            A[j] = A[j] - factor * A[i]  # Rj <- Rj - factor*Ri
            b[j] = b[j] - factor * b[i]  # Apply the same operation on vector b

    # The matrix is now upper triangular, so x can be solved for in O(n^2) 
    # time with backward substituiton (from previous part b)
    return backward_sub(A, b)  

def LU_factorization(A, b):
    """
    Solve the equation Ax = b for x using LU factorization with partial 
    pivoting. Expects a that A is non-singular, and that the dimensions of A 
    and b are compatible (but checks these and throws an exception if not).
    
    Parameters:
    A (ndarray): An matrix of shape (n, n).
    b (ndarray): A vector of shape (n, 1).
    
    Returns:
    x (ndarray): The solution vector of shape (n, 1).
    L (ndarray): The lower triangular matrix of shape (n, n).
    U (ndarray): The upper triangular matrix of shape (n, n).
    P (ndarray): The permutation matrix of shape (n, n).
    """    
    tol = 1e-12                  # Tolerance to avoid floating point issues
    n = A.shape[0]               # Size of matrix A

    if A.shape[1] != n:          # Check if A is square
        raise ValueError("Matrix A must be square.")
    if b.shape[0] != n:          # Check if dimensions of b are compatible
        raise ValueError("Vector b must have compatible dimensions with matrix A.")

    L = np.eye(n)                # Initialize L as identity matrix
    P = np.eye(n)                # Initialize P as identity matrix

    # Essentially apply gaussian elimination with partial pivoting to A, with
    # the added step of building the lower triangular matrix L along the way.
    # Whenever rows are swapped in A, the same row swap is applied to P. 
    for i in range(n - 1):       

        p = i + np.argmax(np.abs(A[i:, i]))

        if np.abs(A[p, i]) <= tol:
                raise ValueError("Matrix is singular.")
        # If the current row is not the one with the greatest pivot element,
        # do partial pivoting by swapping row i with that row.  
        if p != i:
                A[[i, p], :] = A[[p, i], :]
            # Apply same row swap to L up to column i to maintain consistency
                L[[i, p], :i] = L[[p, i], :i]
            # Apply same row swap to P
                P[[i, p], :] = P[[p, i], :]

        for j in range(i + 1, n): 
            factor = A[j, i] / A[i, i] 

            A[j] = A[j] - factor * A[i]
            # for Rj <- Rj - factor * Ri, insert L[j, i] = factor
            L[j, i] = factor # Store the factor in L
            
    U = A.copy() # Upper triangular matrix is the modified A after elimination

    # Solve Ly = Pb and Ux = y using O(n^2) forward and backward substitution
    y = forward_sub(L, np.dot(P, b)) 
    x = backward_sub(U, y)

    return x, L, U, P
