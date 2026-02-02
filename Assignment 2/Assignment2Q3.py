#!/usr/bin/env python
# coding: utf-8

# # MECH 309: Assignment 2, Question 3
# 
# Cagri Arslan
# 
# January 30, 2025
# 
# *All work can be found on https://github.com/imported-canuck/MECH-309*

# In[40]:


# Imports
import numpy as np 
from scipy import linalg
import solver 


# *Note:* I wrote all scripts by hand. Therefore, code might not be as concise/clean as a script produced by ChatGPT or taken from StackOverflow. Regardless, scripts should still be fully functional if all **assumptions** stated in docstrings are respected. I trust that this won't be penalized during grading. 

# In[44]:


def forward_sub(A, b):
    """
    Solve the equation Ax = b for x, where A is a lower triangular matrix. Assumes a lower 
    triangular matrix A. Expects a that A is non-singular, and that the dimensions of 
    A and b are compatible (but checks these and throws an exception if not).

    Parameters:
    A (ndarray): A lower triangular matrix of shape (n, n).
    b (ndarray): A vector of shape (n, 1).

    Returns:
    x (ndarray): The solution vector of shape (n, 1).
    """
    n = A.shape[0]           # Size of matrix A
    if A.shape[1] != n:      # Check if A is square
        raise ValueError("Matrix A must be square.")
    if b.shape[0] != n:      # Check if dimensions of b are compatible
        raise ValueError("Vector b must have compatible dimensions with matrix A.")

    x = np.zeros((n, 1))          # Initialize solution vector x

    x[0, 0] = b[0, 0] / A[0, 0]    # First element of x (actually redundant, kept for clarity)

    for i in range(n):       # Loop over each row
        if A[i, i] == 0:     # Check for singularity (zero diagonal element)
            raise ValueError("Matrix is singular.")
        for j in range(i):   
            b[i, 0] -= A[i, j] * x[j, 0]

        x[i, 0] = b[i, 0] / A[i, i]

    return x  

forward_sub(
    np.array([[2, 0, 0], [3, 1, 0], [-2, -1, 3]]),
    np.array([[2], [6], [1]])
            )


# In[46]:


def backward_sub(A, b):
    """
    Solve the equation Ax = b for x, where A is an upper triangular matrix. Assumes an upper 
    triangular matrix A. Expects a that A is non-singular, and that the dimensions of 
    A and b are compatible (but checks these and throws an exception if not).

    Parameters:
    A (ndarray): An upper triangular matrix of shape (n, n).
    b (ndarray): A vector of shape (n, 1).

    Returns:
    x (ndarray): The solution vector of shape (n, 1).
    """
    n = A.shape[0]               # Size of matrix A
    if A.shape[1] != n:          # Check if A is square
        raise ValueError("Matrix A must be square.")
    if b.shape[0] != n:          # Check if dimensions of b are compatible
        raise ValueError("Vector b must have compatible dimensions with matrix A.")

    x = np.zeros((n, 1))              # Initialize solution vector x

    x[-1, 0] = b[-1, 0] / A[-1, -1]    # First element of x

    for i in range(n - 1, -1 , -1):
        if A[i, i] == 0:         # Check for singularity (zero diagonal element)
            raise ValueError("Matrix is singular.")

        for j in range(i + 1, n):
            b[i, 0] -= A[i, j] * x[j, 0]

        x[i, 0] = b[i, 0] / A[i, i]

    return x

backward_sub(np.array([[1, -3, 1], [0, 2, -2], [0, 0, 3]]), np.array([[2], [-2], [6]]))


# In[47]:


def gaussian_elimination(A, b):
    """
    Solve the equation Ax = b for x using Gauss-Jordan elimination. Expects a that A is non-singular, and that the 
    dimensions of A and b are compatible (but checks these and throws an exception if not).

    Parameters:
    A (ndarray): An matrix of shape (n, n).
    b (ndarray): A vector of shape (n, 1).

    Returns:
    x (ndarray): The solution vector of shape (n, 1).
    """    
    n = A.shape[0]               # Size of matrix A
    if A.shape[1] != n:          # Check if A is square
        raise ValueError("Matrix A must be square.")
    if b.shape[0] != n:          # Check if dimensions of b are compatible
        raise ValueError("Vector b must have compatible dimensions with matrix A.")

    for i in range(n):
        if A[i, i] == 0:         # Check for singularity (zero diagonal element)
            raise ValueError("Matrix is singular.")
        for j in range(i + 1, n): 
            factor = A[j, i] / A[i, i] 
            A[j] = A[j] - factor * A[i]
            b[j] = b[j] - factor * b[i]

    print(A)
    print(b)

    backward_sub(A, b)

gaussian_elimination(
            np.array([[1.0, 3.0, 5.0],
                      [3.0, 5.0, 5.0],
                      [5.0, 5.0, 5.0]]),
            np.array([[9.0],
                      [13.0],
                      [15.0]]),
        )

xtrue = linalg.solve(
            np.array([[1.0, 3.0, 5.0],
                      [3.0, 5.0, 5.0],
                      [5.0, 5.0, 5.0]]),
            np.array([[9.0],
                      [13.0],
                      [15.0]]),
        )

print(xtrue)


# In[ ]:


def LU_factorization(A, b):
    """
    Solve the equation Ax = b for x using LU factorization. Expects a that A is non-singular, 
    and that the dimensions of A and b are compatible (but checks these and throws an exception 
    if not).

    Parameters:
    A (ndarray): An matrix of shape (n, n).
    b (ndarray): A vector of shape (n, 1).

    Returns:
    x (ndarray): The solution vector of shape (n, 1).
    L (ndarray): The lower triangular matrix of shape (n, n).
    U (ndarray): The upper triangular matrix of shape (n, n).
    """    
    n = A.shape[0]               # Size of matrix A
    if A.shape[1] != n:          # Check if A is square
        raise ValueError("Matrix A must be square.")
    if b.shape[0] != n:          # Check if dimensions of b are compatible
        raise ValueError("Vector b must have compatible dimensions with matrix A.")

    L = np.eye(n)               # Initialize L as identity matrix

    for i in range(n):
        if A[i, i] == 0:         # Check for singularity (zero diagonal element)
            raise ValueError("Matrix is singular.")

        for j in range(i + 1, n): 
            factor = A[j, i] / A[i, i] 
            A[j] = A[j] - factor * A[i]
            L[j, i] = factor

    U = A.copy()

    y = forward_sub(L, b)
    x = backward_sub(U, y)

    return L, U, x

LU_factorization(
            np.array([[1.0, 3.0, 5.0],
                      [3.0, 5.0, 5.0],
                      [5.0, 5.0, 5.0]]),
            np.array([[9.0],
                      [13.0],
                      [15.0]]),
                )

