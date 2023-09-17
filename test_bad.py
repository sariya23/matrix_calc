import pytest

from matrix import Matrix


data_create_bad = (
    [],
    [[[[]]]],
    [[], [], []]
)

data_det_bad = (
    (Matrix([[1, 2, 3], [2, 3, 4]]).determinate, None),
    (Matrix([[1, 2], [3, 5], [2, 3]]).determinate, None),
)

data_create_str = (
    [['a', 'b'], ['a', 'b']],
    [['a'], ['b']],
    ['kek']
)

data_add_bad = (
    ([[1, 2, 3], [1, 2, 3]], [[1, 2], [1, 2]]),
    ([[1, 2, 3], [1, 2, 3]], [[1]])
)


@pytest.mark.parametrize('matrix', data_create_bad)
def test_create_bad(matrix):
    with pytest.raises(TypeError):
        a = Matrix(matrix)


@pytest.mark.parametrize('matrix_det, expected_det', data_det_bad)
def test_det_bad(matrix_det, expected_det):
    assert matrix_det == expected_det


@pytest.mark.parametrize('matrix', data_create_str)
def test_create_from_string_error(matrix):
    with pytest.raises(TypeError):
        Matrix(matrix)


@pytest.mark.parametrize('matrix1, matrix2', data_add_bad)
def test_add_bad(matrix1, matrix2):
    with pytest.raises(TypeError):
        Matrix(matrix1) + Matrix(matrix2)