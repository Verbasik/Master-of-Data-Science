from typing import List

def count_less_equal(n: int, mid: int) -> int:
    """
    Description:
        Подсчитывает количество чисел, которые меньше или равны mid в матрице n x n.

    Args:
        n: Размер матрицы (n x n)
        mid: Значение, для которого подсчитывается количество чисел

    Returns:
        Количество чисел, которые меньше или равны mid

    Examples:
        >>> count_less_equal(3, 5)
        6
    """
    count = 0
    for i in range(1, min(n, mid) + 1):
        count += min(mid // i, n)
    return count

def find_kth_number(n: int, k: int) -> int:
    """
    Description:
        Находит k-е по величине число в матрице n x n.

    Args:
        n: Размер матрицы (n x n)
        k: Порядковый номер числа

    Returns:
        k-е по величине число в матрице

    Examples:
        >>> find_kth_number(3, 5)
        5
    """
    left, right = 1, n * n
    while left < right:
        mid = (left + right) // 2
        count = count_less_equal(n, mid)
        if count < k:
            left = mid + 1
        else:
            right = mid
    return left

# Основная часть программы
if __name__ == "__main__":
    import sys
    input = sys.stdin.read
    data = input().split()
    n = int(data[0])
    k = int(data[1])
    print(find_kth_number(n, k))
