"""
Jorge Zuniga
zunigjor
ZS 2022/2023

run:
Linux: python3 main.py
Windows: python3 main.py

requirements: numpy
"""
import numpy as np


GAMMAS = [10, 2, 4/5]
ACCURACY = np.float64(pow(10, -6))
DIMENSION = 20
MAX_ITERATIONS = 1000


def init_A_m(gamma):
    """
    Initialize matrix A

    |γ  -1            |
    |-1  γ -1         |
    |   -1  γ  ...    |
    |      ... ... -1 |
    |           -1  γ |

    with 0 on empty places
    """
    A_m = [[np.float64(0) for i in range(DIMENSION)] for j in range(DIMENSION)]
    for i in range(DIMENSION):
        for j in range(DIMENSION):
            if i == j:
                A_m[i][j] = np.float64(gamma)
            if i == j + 1 or j == i + 1:
                A_m[i][j] = np.float64(-1)
    return A_m


def init_b_v(gamma):
    """
    Initialize vector b

    (γ - 1, γ - 2, ..., γ - 2, γ - 1)
    """
    B_v = [np.float64(gamma - 2) for i in range(DIMENSION)]
    B_v[0] = np.float64(gamma - 1)
    B_v[DIMENSION - 1] = np.float64(gamma - 1)
    return B_v


def init_x_0_v():
    """
    Initialize vector x_0
    (0, 0, ..., 0)
    """
    return [np.float64(0) for i in range(DIMENSION)]


def approximation_acc(A_m, b_v, x):
    """
    Calculate stopping accuracy using euclidean norm: ||x||_2 = (sum(x_i)^2)^1/2

    :param A_m:
    :param b_v:
    :param x: Approximate solution

    :return: ||Ax - b||_2 / ||b||_2
    """
    # Ax - b
    Ax_m = np.matmul(A_m, x, dtype=np.float64)
    Ax_minus_b = np.subtract(Ax_m, b_v, dtype=np.float64)
    # sum
    dividend = np.float64(0)
    divisor = np.float64(0)
    for i in range(DIMENSION):
        dividend = dividend + np.power(Ax_minus_b[i], 2, dtype=np.float64)
        divisor = divisor + np.power(b_v[i], 2, dtype=np.float64)
    # ^1/2
    dividend = np.sqrt(dividend)
    divisor = np.sqrt(divisor)
    return np.divide(dividend, divisor)


def converges(q, A_m):
    """
    W = E - Q^-1*A
    ρ(W) := max{|λ|: λ being an eigenvalue of W}

    :return: True if ρ(W) < 1
    :return: False otherwise (diverges)
    """
    W = np.subtract(np.identity(DIMENSION), np.matmul(np.linalg.inv(q), A_m))
    max_abs_eigenval = max(abs(np.linalg.eigvals(W)))
    if np.less(max_abs_eigenval, np.float64(1)):
        return True
    return False


def run_iteration(method, A_m, b_v, x_0_v, accuracy, max_iter):
    """
    Runs iterations with a desired x calculation method.

    :param method: Method to calculate next x.
    :param A_m: Matrix A
    :param b_v: Vector b
    :param x_0_v: Vector x_0_v
    :param accuracy: Accuracy
    :param max_iter: Maximum iterations

    :return: Number of iterations
    :return: "Diverges" if doesn't pass the convergence test
    :return: "Result not found" if max number of iterations reached
    """
    k = 1
    x = []
    while k <= max_iter:
        x = method(A_m, b_v, x_0_v)
        if x is None:
            return "Diverges"
        if np.less(approximation_acc(A_m, b_v, x), accuracy):
            return k
        k = k + 1
        x_0_v = x
    return "Result not found"


def jacobi_x(A_m, b_v, x_prev_v):
    """
    Computing vector x using Jacobi method:
    Q = D

    :param: A_m
    :param: b_v
    :param: x_prev_v Previous calculated x.

    :return: x_k = Q^-1 * ((Q-A)*x_k-1 + b)
    """
    D = np.diagflat(np.diag(A_m))
    Q = D
    if not converges(Q, A_m):
        return None
    return np.matmul(np.linalg.inv(Q), np.add(np.matmul(np.subtract(Q, A_m), x_prev_v), b_v))


def gauss_seidel_x(A_m, b_v, x_prev_v):
    """
    Computing vector x using Gauss-Seidel method:
    Q = D + L

    :param: A_m
    :param: b_v
    :param: x_prev_v Previous calculated x.

    :return: x_k = Q^-1 * ((Q-A)*x_k-1 + b)
    """
    L = np.tril(A_m, -1)
    D = np.diagflat(np.diag(A_m))
    Q = np.add(D, L)
    if not converges(Q, A_m):
        return None
    return np.matmul(np.linalg.inv(Q), np.add(np.matmul(np.subtract(Q, A_m), x_prev_v), b_v))


if __name__ == '__main__':
    print("=" * 50)
    for gamma in GAMMAS:
        print(f"γ = {gamma}")
        A_m = init_A_m(gamma)
        b_v = init_b_v(gamma)
        x_0_v = init_x_0_v()
        result_jacobi = run_iteration(jacobi_x, A_m, b_v, x_0_v, ACCURACY, MAX_ITERATIONS)
        print(f"Jacobi: {result_jacobi}")
        result_gauss_seidel = run_iteration(gauss_seidel_x, A_m, b_v, x_0_v, ACCURACY, MAX_ITERATIONS)
        print(f"Gauss-Seidel: {result_gauss_seidel}")
        print("="*50)
