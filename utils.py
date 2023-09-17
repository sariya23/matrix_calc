from copy import deepcopy


def is_empty(lst):
    if not lst:
        raise TypeError

    for item in lst:
        if isinstance(item, list):
            if not is_empty(item):
                return
        else:
            return

    raise TypeError


def check_is_input_correct(matrix):
    if not isinstance(matrix, list):
        return False

    if not all(isinstance(i, list) for i in matrix):
        return False

    if len(set(map(len, matrix))) != 1:
        return False

    return True


def determinant_recursive(matrix):
    if len(matrix) == len(matrix[0]) == 1:
        return matrix[0][0]

    if len(matrix) != len(matrix[0]):
        return None

    total = 0
    indices = list(range(len(matrix)))

    if len(matrix) == 2 and len(matrix[0]) == 2:
        val = matrix[0][0] * matrix[1][1] - matrix[1][0] * matrix[0][1]
        return val

    for fc in indices:
        As = deepcopy(matrix)
        As = As[1:]
        height = len(As)

        for i in range(height):
            As[i] = As[i][0:fc] + As[i][fc + 1:]

        sign = (-1) ** (fc % 2)
        sub_det = determinant_recursive(As)

        total += sign * matrix[0][fc] * sub_det

    return total


def get_size_matrix(matrix):
    return len(matrix), len(matrix[0])


def add_matrix(matrix1, matrix2):
    R, C = get_size_matrix(matrix1)
    new_matrix = [[0] * C for _ in range(R)]

    for i in range(R):
        for j in range(C):
            new_matrix[i][j] = matrix1[i][j] + matrix2[i][j]

    return new_matrix


def sub_matrix(matrix1, matrix2):
    R, C = get_size_matrix(matrix1)
    new_matrix = [[0] * C for _ in range(R)]

    for i in range(R):
        for j in range(C):
            new_matrix[i][j] = matrix1[i][j] - matrix2[i][j]

    return new_matrix


def mul_matrix(matrix1, matrix2):
    C2 = len(matrix2[0])
    R1 = len(matrix1)
    R2 = len(matrix2)
    new_matrix = [[0] * C2 for _ in range(R1)]

    for i in range(R1):
        for j in range(C2):
            for k in range(R2):
                new_matrix[i][j] += matrix1[i][k] * matrix2[k][j]

    return new_matrix


def mul_number_to_matrix(number, matrix):
    R, C = get_size_matrix(matrix)
    new_matrix = [[0] * C for _ in range(R)]

    for i in range(R):
        for j in range(C):
            new_matrix[i][j] = matrix[i][j] * number

    return new_matrix


def pow_matrix(power, matrix):
    if len(matrix) != len(matrix[0]):
        raise TypeError('It is not a square matrix!')

    new_matrix = deepcopy(matrix)

    for i in range(power - 1):
        new_matrix = mul_matrix(new_matrix, matrix)

    return new_matrix


def transpose_matrix(matrix):
    res = [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]
    return res


