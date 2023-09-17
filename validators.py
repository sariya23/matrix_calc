class ValidMatrix:
    def __set_name__(self, cls, attr):
        self._attr = attr

    def __get__(self, instance, owner):
        if self._attr in instance.__dict__:
            return instance.__dict__[self._attr]
        else:
            raise AttributeError('Атрибут не существует')

    def __set__(self, instance, value):
        if isinstance(value, list):
            self.__is_empty_recursive(value)
            instance.__dict__[self._attr] = value
        else:
            raise ValueError('Некорректное значение')

    def __is_empty_recursive(self, lst):
        if not lst:
            raise TypeError('List is empty')

        for item in lst:
            if isinstance(item, list):
                if not self.__is_empty_recursive(item):
                    return False
            else:
                return False

        raise TypeError('List is empty')
