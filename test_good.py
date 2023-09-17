import pytest
import sympy
import numpy as np

from matrix import Matrix


data_det_good = (
    (Matrix([[2, 3], [-1, 4]]), sympy.det(sympy.Matrix([[2, 3], [-1, 4]]))),
    (Matrix([[1, -3], [2, 5]]), sympy.det(sympy.Matrix([[1, -3], [2, 5]]))),
    (Matrix([[3, 0], [1, 9]]), sympy.det(sympy.Matrix([[3, 0], [1, 9]]))),
    (Matrix([[2, 1, 5], [3, -1, 2], [0, 4, 2]]), sympy.det(sympy.Matrix([[2, 1, 5], [3, -1, 2], [0, 4, 2]]))),
    (Matrix([[3, -2, 5, 1], [4, 0, 2, 3], [-1, 2, 0, 4], [2, 1, 3, -1]]), sympy.det(sympy.Matrix([[3, -2, 5, 1], [4, 0, 2, 3], [-1, 2, 0, 4], [2, 1, 3, -1]]))),
    (Matrix([[3, -2, 5, 1, 2], [4, 0, 2, 3, -1], [-1, 2, 0, 4, 5], [2, 1, 3, -1, 0], [1, -3, 2, 4, 1]]), sympy.det(sympy.Matrix([[3, -2, 5, 1, 2], [4, 0, 2, 3, -1], [-1, 2, 0, 4, 5], [2, 1, 3, -1, 0], [1, -3, 2, 4, 1]]))),
)

data_transpose_good = (
    (Matrix([[1]]).t, np.transpose(np.array([[1]]))),
    (Matrix([[2, 3], [-1, 4]]).t, np.transpose(np.array([[2, 3], [-1, 4]]))),
    (Matrix([[1, 2, 3], [4, 5, 6]]).t, np.transpose(np.array([[1, 2, 3], [4, 5, 6]]))),
    (Matrix([[1, -3], [2, 5]]).t, np.transpose(np.array([[1, -3], [2, 5]]))),
    (Matrix([[2, 1, 5], [3, -1, 2], [0, 4, 2]]).t, np.transpose(np.array([[2, 1, 5], [3, -1, 2], [0, 4, 2]]))),
    (Matrix([[3, -2, 5, 1, 2], [4, 0, 2, 3, -1], [-1, 2, 0, 4, 5], [2, 1, 3, -1, 0], [1, -3, 2, 4, 1]]).t, np.transpose(np.array([[3, -2, 5, 1, 2], [4, 0, 2, 3, -1], [-1, 2, 0, 4, 5], [2, 1, 3, -1, 0], [1, -3, 2, 4, 1]])))
)

data_add_good = (
    (Matrix([[2, 3], [-1, 4]]), Matrix([[1, -3], [2, 5]]), np.array([[2, 3], [-1, 4]]) + np.array([[1, -3], [2, 5]])),
    (Matrix([[2, 1, 5], [3, -1, 2], [0, 4, 2]]), Matrix([[1, 0, -1], [2, 1, -2], [0, 0, 0]]), np.array([[2, 1, 5], [3, -1, 2], [0, 4, 2]]) + np.array([[1, 0, -1], [2, 1, -2], [0, 0, 0]])),
    (Matrix([[3, -2, 5, 1, 2], [4, 0, 2, 3, -1], [-1, 2, 0, 4, 5]]), Matrix([[-1, 0, 2, 1, 3], [0, 1, -2, 4, -1], [3, -2, 1, 0, 1]]), np.array([[3, -2, 5, 1, 2], [4, 0, 2, 3, -1], [-1, 2, 0, 4, 5]]) + np.array([[-1, 0, 2, 1, 3], [0, 1, -2, 4, -1], [3, -2, 1, 0, 1]]))
)

data_sub_good = (
    (Matrix([[2, 3], [-1, 4]]), Matrix([[1, -3], [2, 5]]), np.array([[2, 3], [-1, 4]]) - np.array([[1, -3], [2, 5]])),
    (Matrix([[2, 1, 5], [3, -1, 2], [0, 4, 2]]), Matrix([[1, 0, -1], [2, 1, -2], [0, 0, 0]]), np.array([[2, 1, 5], [3, -1, 2], [0, 4, 2]]) - np.array([[1, 0, -1], [2, 1, -2], [0, 0, 0]])),
    (Matrix([[3, -2, 5, 1, 2], [4, 0, 2, 3, -1], [-1, 2, 0, 4, 5]]), Matrix([[-1, 0, 2, 1, 3], [0, 1, -2, 4, -1], [3, -2, 1, 0, 1]]), np.array([[3, -2, 5, 1, 2], [4, 0, 2, 3, -1], [-1, 2, 0, 4, 5]]) - np.array([[-1, 0, 2, 1, 3], [0, 1, -2, 4, -1], [3, -2, 1, 0, 1]]))
)

data_mul_good = (
    (Matrix([[2, 3, 1], [-1, 4, 2]]), Matrix([[1, -3], [2, 5], [3, 0]]), np.matmul(np.array([[2, 3, 1], [-1, 4, 2]]), np.array([[1, -3], [2, 5], [3, 0]]))),
    (Matrix([[0, 1, 4], [4, -2, 1]]), Matrix([[5, 3], [2, 0], [-1, 2]]), np.matmul(np.array([[0, 1, 4], [4, -2, 1]]), np.array([[5, 3], [2, 0], [-1, 2]]))),
    (Matrix([[7, -2], [3, 6], [1, 2]]), Matrix([[2, 1], [-1, 0]]), np.matmul(np.array([[7, -2], [3, 6], [1, 2]]), np.array([[2, 1], [-1, 0]]))),
    (Matrix([[1, 2], [3, 5]]), Matrix([[6, 7], [1, 5]]), np.matmul(np.array([[1, 2], [3, 5]]), np.array([[6, 7], [1, 5]])))
)


data_pow_good = (
    (Matrix([[2, 3], [-1, 4]]), 2, np.linalg.matrix_power(np.array([[2, 3], [-1, 4]]), 2)),
    (Matrix([[1, -3, 2], [2, 5, 1], [2, 3, 4]]), 3, np.linalg.matrix_power(np.array([[1, -3, 2], [2, 5, 1], [2, 3, 4]]), 3)),
    (Matrix([[2, 1, 5, 3], [3, -1, 2, 0], [3, 4, 1, 3], [4, 5, 6, 7]]), 4, np.linalg.matrix_power(np.array([[2, 1, 5, 3], [3, -1, 2, 0], [3, 4, 1, 3], [4, 5, 6, 7]]), 4))
)


@pytest.mark.parametrize('matrix, expected_t', data_transpose_good)
def test_transpose(matrix, expected_t):
    assert matrix.matrix_list == expected_t.tolist()


@pytest.mark.parametrize('matrix, expected_det', data_det_good)
def test_det(matrix, expected_det):
    assert matrix.determinate == expected_det


@pytest.mark.parametrize('matrix1, matrix2, expected_sum', data_add_good)
def test_add(matrix1, matrix2, expected_sum):
    assert (matrix1 + matrix2).matrix_list == expected_sum.tolist()


@pytest.mark.parametrize('matrix1, matrix2, expected_sub', data_sub_good)
def test_sub(matrix1, matrix2, expected_sub):
    assert (matrix1 - matrix2).matrix_list == expected_sub.tolist()


@pytest.mark.parametrize('matrix1, matrix2, expected_mul', data_mul_good)
def test_mul(matrix1, matrix2, expected_mul):
    assert (matrix1 * matrix2).matrix_list == expected_mul.tolist()


@pytest.mark.parametrize('matrix, power, expected_res', data_pow_good)
def test_pow(matrix, power, expected_res):
    assert (matrix ** power).matrix_list == expected_res.tolist()
