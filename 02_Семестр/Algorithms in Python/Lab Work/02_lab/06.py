# Импорт стандартных библиотек
import sys
from typing import List, Tuple


def solve() -> None:
    """
    Description:
        Решает задачу поиска наибольшего подотрезка массива, среднее значение которого 
        не меньше заданного порога, с использованием метода бинарного поиска.

    Args:
        Данные считываются из `sys.stdin`, включая:
        - Первую строку: два целых числа `n` (размер массива) и `d` (минимальная длина подотрезка).
        - Вторую строку: массив `a` из `n` целых чисел.

    Returns:
        Выводит в `sys.stdout` два числа — индексы (1-индексация) левого и правого концов 
        найденного подотрезка.
    
    Raises:
        ValueError: Если входные данные некорректны.

    Examples:
        Вход:
            5 2
            1 2 3 4 5
        Выход:
            4 5
    """

    # Чтение первой строки с n и d
    line = sys.stdin.readline()
    if not line:
        return
    
    n, d = map(int, line.strip().split())

    # Чтение массива чисел
    a = list(map(int, sys.stdin.readline().strip().split()))

    # Границы бинарного поиска (диапазон возможных средних значений)
    lo, hi = 0.0, 100.0

    # Изначально лучший найденный отрезок — [1, d]
    best_l, best_r = 1, d

    # Префиксные суммы для преобразованного массива
    prefix: List[float] = [0.0] * (n + 1)

    # Бинарный поиск с 60 итерациями (точность достаточна)
    for _ in range(60):
        mid = (lo + hi) / 2.0  # Среднее значение

        prefix[0] = 0.0

        # Вычисляем префиксные суммы для массива, преобразованного по `a[i] - mid`
        for i in range(1, n + 1):
            prefix[i] = prefix[i - 1] + (a[i - 1] - mid)

        # Флаг, обозначающий, нашли ли мы подходящий подотрезок
        feasible = False

        # Минимальная префиксная сумма и ее индекс
        min_prefix = 0.0
        min_index = 0

        # Перебираем возможные правые границы подотрезка
        for i in range(d, n + 1):
            # Обновляем минимальную префиксную сумму
            if prefix[i - d] < min_prefix:
                min_prefix = prefix[i - d]
                min_index = i - d

            # Проверяем, существует ли подотрезок длины >= d с неотрицательной суммой
            if prefix[i] - min_prefix >= 0:
                feasible = True
                current_l = min_index + 1  # Переход к 1-индексации
                current_r = i
                break

        # Обновляем границы бинарного поиска
        if feasible:
            lo = mid
            best_l, best_r = current_l, current_r
        else:
            hi = mid

    # Выводим найденный подотрезок
    sys.stdout.write(f"{best_l} {best_r}\n")