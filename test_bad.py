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


@pytest.mark.parametrize('matrix', data_create_bad)
def test_create_bad(matrix):
    with pytest.raises(TypeError):
        a = Matrix(matrix)


@pytest.mark.parametrize('matrix_det, expected_det', data_det_bad)
def test_det_bad(matrix_det, expected_det):
    assert matrix_det == expected_det
