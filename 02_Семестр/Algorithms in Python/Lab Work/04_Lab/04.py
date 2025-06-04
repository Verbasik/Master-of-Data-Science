# Асимптотическая сложность по времени - О(TODO)
# Асимптотическая сложность по памяти - O(TODO)

# ВАШ КОД
from typing import List

def solve_sliding_window_length_sum() -> None:
    """
    Description:
    ---------------
        Вычисляет сумму длин всех подмассивов, сумма элементов которых не превышает заданное значение s.

    Args:
    ---------------
        None (ввод осуществляется через стандартный ввод)

    Returns:
    ---------------
        None (вывод осуществляется через стандартный вывод)

    Examples:
    ---------------
        >>> solve_sliding_window_length_sum()
        Ввод:
        5 6
        1 2 3 4 5
        Вывод:
        25
    """
    n, s = map(int, input().split())
    a = list(map(int, input().split()))

    total_length_sum = 0
    left = 0
    current_sum = 0

    # Проходим по массиву справа налево
    for right in range(n):
        current_sum += a[right]

        # Поддерживаем текущую сумму не больше s
        while current_sum > s:
            current_sum -= a[left]
            left += 1

        # Вычисляем количество подмассивов, заканчивающихся на текущем элементе
        length_count = (right - left + 1)
        if length_count > 0:
            # Сумма длин всех подмассивов, заканчивающихся на текущем элементе
            sum_of_lengths_for_right = (length_count * (length_count + 1)) // 2
            total_length_sum += sum_of_lengths_for_right

    print(total_length_sum)

if __name__ == "__main__":
    solve_sliding_window_length_sum()