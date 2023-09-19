from utils import determinant_recursive,\
    add_matrix, sub_matrix, mul_matrix, mul_number_to_matrix,\
    pow_matrix, transpose_matrix, inverse_matrix, do_decimal

from validators import ValidMatrix
import numpy as np
from typing import Type


class Matrix:
    """
    Класс представляет собой матрицу.
    """
    matrix = ValidMatrix()

    def __init__(self, matrix_list):
        self.matrix = do_decimal(matrix_list)
        self._determinate = determinant_recursive(matrix_list)
        self._size = (len(self.matrix), len(self.matrix[0]))
        self._t = transpose_matrix(matrix_list)
        self._matrix_list = do_decimal(matrix_list)

    def __add__(self, other) -> 'Matrix':
        """
        Сложение матриц.

        :param other: Экземпляр класса Matrix.
        :return: Результат сложения двух матриц в виде экземпляра класса Matrix.
        """
        if isinstance(other, Matrix):
            if other.size == self.size:
                return Matrix(add_matrix(self.matrix, other.matrix))
            raise TypeError('Size is not equal')
        return NotImplemented

    def __sub__(self, other):
        """
        Вычитание матриц.

        :param other: Экземпляр класса Matrix.
        :return: Результат вычитания двух матриц в виде экземпляра класса Matrix.
        """
        if isinstance(other, Matrix):
            if other.size == self.size:
                return Matrix(sub_matrix(self.matrix, other.matrix))
            raise TypeError('Size is not equal')
        return NotImplemented

    def __mul__(self, other):
        """
        Умножение матриц.

        :param other: Экземпляр класса Matrix.
        :return: Результат перемножения двух матриц в виде экземпляра класса Matrix.
        """
        if isinstance(other, Matrix):
            if self.size[1] == other.size[0]:
                return Matrix(mul_matrix(self.matrix, other.matrix))
            raise TypeError('Sizes is not equal')
        elif isinstance(other, int | float):
            return Matrix(mul_number_to_matrix(other, self.matrix))
        return NotImplemented

    def __pow__(self, power):
        """
        Возведение в степень. Если степень равна -1, то
        вернется обратная матрица.

        :param power: Значение степени.
        :return: Результат возведения матрицы в степень в виде экземпляра класса Matrix.
        """
        if isinstance(power, int | float):
            if power == -1:
                return Matrix(inverse_matrix(self.matrix))
            return Matrix(pow_matrix(power, self.matrix))
        return NotImplemented

    def __str__(self) -> str:
        res = ''

        for i in range(len(self.matrix)):
            for j in range(len(self.matrix[i])):
                res += str(self.matrix[i][j]) + ' '
            res += '\n'

        return res

    def __len__(self) -> int:
        return len(self.matrix)

    def __eq__(self, other) -> bool:
        if isinstance(other, Matrix):
            return self.matrix == other.matrix
        return NotImplemented

    @classmethod
    def from_string(cls, matrix_string):
        pass

    @property
    def determinate(self) -> int | float:
        return self._determinate

    @property
    def size(self) -> tuple[int, int]:
        return self._size

    @property
    def t(self) -> 'Matrix':
        self._t = Matrix(self._t)
        return self._t

    @property
    def matrix_list(self) -> list[list, ...]:
        return self._matrix_list


if __name__ == '__main__':
    a = Matrix([[2, 3], [-1, 4]])

    print((a ** -1).matrix_list == np.linalg.inv(np.array([[2, 3], [-1, 4]])).tolist())
    print((a ** -1).matrix_list)
    print(np.linalg.inv(np.array([[2, 3], [-1, 4]])).tolist())