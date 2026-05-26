#!/usr/bin/env python3

"""
Module for matrix inverse calculation.
"""


def inverse(matrix):
    """
    Calculate the inverse matrix of a square matrix.

    Args:
        matrix: A list of lists representing a square matrix.

    Returns:
        The inverse matrix as a list of lists, or None if matrix is singular.

    Raises:
        TypeError: If matrix is not a list of lists.
        ValueError: If matrix is not a non-empty square matrix.
    """

    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    if len(matrix) == 0:
        raise ValueError("matrix must be a list of lists")

    rows = len(matrix)
    cols = len(matrix[0])

    if rows == 1 and cols == 0:
        raise ValueError("matrix must be a non-empty square matrix")

    if rows != cols:
        raise ValueError("matrix must be a non-empty square matrix")

    if not all(len(row) == cols for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    adjugate_matrix = adjugate(matrix)
    det = determinant(matrix)

    # Check if matrix is singular
    if det == 0:
        return None

    # Calculate inverse = adjugate / determinant
    inverse_matrix = []
    for i in range(rows):
        row = []
        for j in range(cols):
            row.append(adjugate_matrix[i][j] / det)
        inverse_matrix.append(row)

    return inverse_matrix


def adjugate(matrix):
    """
    Calculate the adjugate matrix of a square matrix.

    Args:
        matrix: A list of lists representing a square matrix.

    Returns:
        The adjugate matrix as a list of lists.

    Raises:
        TypeError: If matrix is not a list of lists.
        ValueError: If matrix is not a non-empty square matrix.
    """

    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    if len(matrix) == 0:
        raise ValueError("matrix must be a list of lists")

    rows = len(matrix)
    cols = len(matrix[0])

    if rows == 1 and cols == 0:
        raise ValueError("matrix must be a non-empty square matrix")

    if rows != cols:
        raise ValueError("matrix must be a non-empty square matrix")

    if not all(len(row) == cols for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    cofactor_matrix = cofactor(matrix)

    # Transpose the cofactor matrix to get the adjugate matrix
    adjugate_matrix = []
    for i in range(rows):
        row = []
        for j in range(cols):
            row.append(cofactor_matrix[j][i])  # Transpose: swap i and j
        adjugate_matrix.append(row)

    return adjugate_matrix


def cofactor(matrix):
    """
    Calculate the cofactor matrix of a square matrix.

    Args:
        matrix: A list of lists representing a square matrix.

    Returns:
        The cofactor matrix as a list of lists.

    Raises:
        TypeError: If matrix is not a list of lists.
        ValueError: If matrix is not a non-empty square matrix.
    """

    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    if len(matrix) == 0:
        raise ValueError("matrix must be a list of lists")

    rows = len(matrix)
    cols = len(matrix[0])

    if rows == 1 and cols == 0:
        raise ValueError("matrix must be a non-empty square matrix")

    if rows != cols:
        raise ValueError("matrix must be a non-empty square matrix")

    if not all(len(row) == cols for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    minor_matrix = minor(matrix)
    n = len(matrix)

    cofactor_matrix = []
    for i in range(n):
        row = []
        for j in range(n):
            sign = 1 if (i + j) % 2 == 0 else -1
            cofactor_value = sign * minor_matrix[i][j]
            row.append(cofactor_value)
        cofactor_matrix.append(row)

    return cofactor_matrix


def determinant(matrix):
    """
    Calculate the determinant of a square matrix.

    Args:
        matrix: A list of lists representing a square matrix.

    Returns:
        The determinant of the matrix as an integer or float.

    Raises:
        TypeError: If matrix is not a list of lists.
        ValueError: If matrix is not square or rows have inconsistent lengths.
    """
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")
    if matrix == []:
        raise ValueError("matrix must be a square matrix")
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    if len(matrix) == 1 and len(matrix[0]) == 0:
        return 1

    rows = len(matrix)
    if rows == 0:
        return 1

    cols = len(matrix[0])

    if rows != cols:
        raise ValueError("matrix must be a square matrix")

    if not all(len(row) == cols for row in matrix):
        raise ValueError("matrix must be a square matrix")

    if rows == 1:
        return matrix[0][0]

    if rows == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    for j in range(cols):
        # Create submatrix by removing first row and column j
        submatrix = []
        for i in range(1, rows):
            row = []
            for k in range(cols):
                if k != j:
                    row.append(matrix[i][k])
            submatrix.append(row)

        sign = 1 if j % 2 == 0 else -1

        det += sign * matrix[0][j] * determinant(submatrix)

    return det


def minor(matrix):
    """
    Calculate the minor matrix of a square matrix.

    Args:
        matrix: A list of lists representing a square matrix.

    Returns:
        The minor matrix as a list of lists.

    Raises:
        TypeError: If matrix is not a list of lists.
        ValueError: If matrix is not a non-empty square matrix.
    """

    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    if len(matrix) == 0:
        raise ValueError("matrix must be a list of lists")

    rows = len(matrix)
    cols = len(matrix[0])

    if rows == 1 and cols == 0:
        raise ValueError("matrix must be a non-empty square matrix")
    if rows != cols:
        raise ValueError("matrix must be a non-empty square matrix")
    if not all(len(row) == cols for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    if rows == 1:
        return [[1]]

    minor_matrix = []

    for i in range(rows):
        minor_row = []
        for j in range(cols):
            # Create submatrix by removing row i and column j
            submatrix = []
            for r in range(rows):
                if r == i:
                    continue
                sub_row = []
                for c in range(cols):
                    if c == j:
                        continue
                    sub_row.append(matrix[r][c])
                submatrix.append(sub_row)
            minor_row.append(determinant(submatrix))

        minor_matrix.append(minor_row)

    return minor_matrix
