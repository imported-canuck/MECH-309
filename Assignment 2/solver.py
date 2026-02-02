# %%
import numpy as np
from scipy import linalg
from Assignment2Q3 import *

# ----------------------------
# STUDENT: choose a test case
# ----------------------------
def load_test_case(case: str = "2x2") -> tuple[np.ndarray, np.ndarray]:
    """
    Return (A, b) for a selected test case.

    Parameters
    ----------
    case : str
        Name of the test case.

    Returns
    -------
    A : np.ndarray
        Matrix of shape (n, n)
    b : np.ndarray
        RHS of shape (n, 1)
    """
    cases: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "2x2": (
            np.array([[6.0, -1.0],
                      [-1.0, 3.0]]),
            np.array([[4.0],
                      [5.0]]),
        ),
        "3x3": (
            np.array([[1.0, 3.0, 5.0],
                      [3.0, 5.0, 5.0],
                      [5.0, 5.0, 5.0]]),
            np.array([[9.0],
                      [13.0],
                      [15.0]]),
        ),
        "5x5": (
            np.array([[-2.0, 1.0, 1.0, -1.0, 6.0],
                      [4.0, 3.0, 3.0, -3.0, 9.0],
                      [8.0, 7.0, 10.0, -4.0, -2.0],
                      [1.0, -9.0, 3.0, 4.0, 1.0],
                      [10.0, -1.0, 1.0, -4.0, 10.0]]),
            np.array([[9.0],
                      [28.0],
                      [38.0],
                      [13.0],
                      [41.0]]),
        ),
    }

    if case not in cases:
        raise ValueError(f"Unknown case '{case}'. Available: {list(cases)}")

    A, b = cases[case]
    return A, b

def print_solution_report(A: np.ndarray, b: np.ndarray, x: np.ndarray, label: str) -> None:
    """
    Print a small report: residual and comparison to SciPy.
    """
    x = np.asarray(x, dtype=float).reshape(-1, 1)

    r = b - A @ x
    print(f"--- {label} ---")
    print("x =\n", x)
    print("||r||_2 =", float(linalg.norm(r, 2)))
    print("")

# %%

# ----------------------------
# STUDENT: your solver goes here
# ----------------------------
def my_solver_template(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    gaussian_elimination(A, b)


# %%
def main():
    for CASE in ["2x2", "3x3", "5x5"]:
        print(f"\n=== Test case: {CASE} ===")

        A, b = load_test_case(CASE)

        # TODO: implement your solver
        # x = my_solver_template(A, b)
        # print_solution_report(A, b, x, label="My solver")

        # Reference solution (allowed for checking)
        x_ref = linalg.solve(A, b)
        print_solution_report(A, b, x_ref, label="Reference (scipy.linalg.solve)")

# %%
if __name__ == "__main__":
    main()
