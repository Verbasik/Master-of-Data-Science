from typing import List

def count_less_equal(a: List[int], b: List[int], mid: int) -> int:
    """
    Description:
        Подсчитывает количество пар (ai, bj), таких что ai + bj <= mid.

    Args:
        a: Первый отсортированный массив
        b: Второй отсортированный массив
        mid: Значение, для которого подсчитывается количество пар

    Returns:
        Количество пар, сумма которых меньше или равна mid

    Examples:
        >>> count_less_equal([1, 2, 3], [4, 5, 6], 5)
        3
    """
    count = 0
    j = len(b) - 1  # Указатель для массива b
    for ai in a:
        while j >= 0 and ai + b[j] > mid:
            j -= 1
        count += j + 1
    return count

def find_kth_sum(n: int, k: int, a: List[int], b: List[int]) -> int:
    """
    Description:
        Находит k-ю по величине сумму пар элементов из массивов a и b.

    Args:
        n: Размер массивов a и b
        k: Порядковый номер суммы
        a: Первый массив
        b: Второй массив

    Returns:
        k-я по величине сумма пар элементов

    Examples:
        >>> find_kth_sum(3, 5, [1, 2, 3], [4, 5, 6])
        7
    """
    a.sort()
    b.sort()
    left, right = a[0] + b[0], a[-1] + b[-1]
    while left < right:
        mid = (left + right) // 2
        count = count_less_equal(a, b, mid)
        if count < k:
            left = mid + 1
        else:
            right = mid
    return left

# Чтение входных данных
n, k = map(int, input().split())
a = list(map(int, input().split()))
b = list(map(int, input().split()))

# Поиск k-й суммы
print(find_kth_sum(n, k, a, b))