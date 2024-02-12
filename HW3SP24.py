# the AI bot Claude was used to assist in writing the program and to help me understand the methods being used.
# Then, ChatGPT was used for assistance in debugging the code. 


import math
from math import erfc, sqrt


def is_sym_pos_def(A):
    """
    Check if a matrix is symmetric and positive definite.

    Args:
    A (list of lists): The input matrix.

    Returns:
    bool: True if the matrix is symmetric positive definite, False otherwise.
    """
    n = len(A)
    if not square_matrix(A, n):
        return False
    for i in range(n):
        for j in range(n):
            if A[i][j] != A[j][i]:
                return False
    eigenvalues = eigenvals(A)
    return all(ev > 0 for ev in eigenvalues)


def square_matrix(A, n):
    """
    Check if a matrix is square.

    Args:
    A (list of lists): The input matrix.
    n (int): The size of the matrix.

    Returns:
    bool: True if the matrix is square, False otherwise.
    """
    for row in A:
        if len(row) != n:
            return False
    return True


def calculate_det(A):
    """
    Calculate the determinant of a matrix recursively.

    Args:
    A (list of lists): The input matrix.

    Returns:
    float: The determinant of the matrix.
    """
    n = len(A)
    if n == 1:
        return A[0][0]
    else:
        det = 0
        for j in range(n):
            M = [[A[row][col] for col in range(n) if col != j] for row in range(n-1)]
            sign = (-1)**(j)
            det += sign * A[0][j] * calculate_det(M)
        return det


def eigenvals(A):
    """
    Calculate eigenvalues of a matrix.

    Args:
    A (list of lists): The input matrix.

    Returns:
    list: A list of eigenvalues.
    """
    eigenvalues = []
    n = len(A)
    for i in range(n):
        M = [[A[row][col] if row != i and col != i else 0 for col in range(n)] for row in range(n)]
        eigenvalues.append(calculate_det(M))
    return eigenvalues


def cholesky(A):
    """
    Perform Cholesky decomposition of a symmetric positive definite matrix.

    Args:
    A (list of lists): The input matrix.

    Returns:
    list of lists: The lower triangular matrix L of the Cholesky decomposition.
    """
    n = len(A)
    L = [[0.0] * n for i in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] ** 2 for k in range(j))
            if i == j:
                L[i][j] = math.sqrt(A[i][i] - s)
            else:
                L[i][j] = (1.0 / L[j][j] * (A[i][j] - s))
    return L


def solve_tri(A, b, lower=True):
    """
    Solve a triangular linear system.

    Args:
    A (list of lists): The triangular matrix.
    b (list): The right-hand side vector.
    lower (bool): True if A is lower triangular, False if upper triangular.

    Returns:
    list: The solution vector x.
    """
    n = len(A)
    x = [0]*n
    if lower:
        for i in range(n):
            s = sum(A[i][j]*x[j] for j in range(i))
            if A[i][i] != 0:
                x[i] = (b[i] - s) / A[i][i]
            else:
                x[i] = b[i] - s
    else:
        for i in range(n-1, -1, -1):
            s = sum(A[i][j]*x[j] for j in range(i+1, n))
            if A[i][i] != 0:
                x[i] = (b[i] - s) / A[i][i]
            else:
                x[i] = b[i] - s
    return x


def doolittle(A):
    """
    Perform LU factorization using the Doolittle method.

    Args:
    A (list of lists): The input matrix.

    Returns:
    tuple: A tuple containing the lower triangular matrix L and the upper triangular matrix U.
    """
    n = len(A)
    L = [[0.0] * n for i in range(n)]
    U = [[0.0] * n for i in range(n)]
    for i in range(n):
        U[i][i] = A[i][i]
        for j in range(i):
            s = sum(L[i][k] * U[k][i] for k in range(j))
            U[i][i] -= s
        for j in range(i + 1, n):
            s = sum(L[j][k] * U[k][i] for k in range(i))
            L[j][i] = (A[j][i] - s) / U[i][i]
    return L, U


A1 = [[5, -2, 2], [-2, 8, -1], [2, -1, 7]]
b1 = [1, -2, 1]

A2 = [[3, 1, 1], [1, 5, 1], [1, 1, 4]]
b2 = [5, 6, 5]

print("Problem 1:")
try:
    if is_sym_pos_def(A1):
        L = cholesky(A1)
        y = solve_tri(L, b1, lower=True)
        x = solve_tri(list(map(list, zip(*L))), y)
        print("Using Cholesky method:")
        print("x =", x)
    else:
        L, U = doolittle(A1)
        y = solve_tri(L, b1, lower=True)
        x = solve_tri(U, y)
        print("Using Doolittle LU factorization:")
        print("x =", x)
except Exception as e:
    print(f"An error occurred: {e}")

print("\nProblem 2:")
try:
    if is_sym_pos_def(A2):
        L = cholesky(A2)
        y = solve_tri(L, b2, lower=True)
        x = solve_tri(list(map(list, zip(*L))), y)
        print("Using Cholesky method:")
        print("x =", x)
    else:
        L, U = doolittle(A2)
        y = solve_tri(L, b2, lower=True)
        x = solve_tri(U, y)
        print("Using Doolittle LU factorization:")
        print("x =", x)
except Exception as e:
    print(f"An error occurred: {e}")


def t_dist_cdf(df, z):
    """
    Compute the cumulative distribution function of the t-distribution.

    Args:
    df (int): Degrees of freedom.
    z (float): The z-value.

    Returns:
    float: The probability.
    """
    return 1 - 0.5 * erfc(abs(z) / sqrt(df) / sqrt(2))


def main():
    # Coefficient matrix A
    A = [[1, -1, 3, 2],
         [4, 2, 4, 0]]

    # Constant vector b
    b = [15, 20]

    print("System of Equations:")
    for i in range(len(b)):
        eq_str = " + ".join([f"{A[i][j]}x{j + 1}" for j in range(len(A[i]))])
        print(eq_str, "=", b[i])

    print()

    # Solve the system using LU decomposition or other methods
    solution, method = decompose_matrix(A)
    print(f"Used method: {method}")
    print_solution_vector(solution)


if __name__ == "__main__":
    try:
        print("Enter degrees of freedom:")
        df = int(input())

        print("Enter z value:")
        z = float(input())

        probability = t_dist_cdf(df, z)

        print(f"T-distribution CDF with {df} degrees of freedom and z = {z}: {probability:.4f}")
    except ValueError:
        print("Invalid input. Please enter numeric values.")
    except Exception as e:
        print(f"An error occurred: {e}")
