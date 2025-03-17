import bisect
from typing import List

def solve_optimized_binary_search() -> None:
    """
    Description:
    ---------------
        Вычисляет количество пар индексов (i, j), таких что d[j] >= d[i] + r + 1,
        где i < j. Использует бинарный поиск для оптимизации.

    Args:
    ---------------
        None (ввод осуществляется через стандартный ввод)

    Returns:
    ---------------
        None (вывод осуществляется через стандартный вывод)

    Examples:
    ---------------
        >>> solve_optimized_binary_search()
        Ввод:
        5 2
        1 3 5 7 9
        Вывод:
        6
    """
    n, r = map(int, input().split())
    d = list(map(int, input().split()))

    count = 0

    # Проходим по всем элементам, кроме последнего
    for i in range(n - 1):
        required_distance = d[i] + r + 1

        # Используем бинарный поиск для нахождения первого элемента,
        # который удовлетворяет условию
        j_index = bisect.bisect_left(d, required_distance, i + 1, n)

        # Если такой элемент найден, увеличиваем счётчик
        if j_index < n:
            count += (n - j_index)

    print(count)

if __name__ == "__main__":
    solve_optimized_binary_search()
