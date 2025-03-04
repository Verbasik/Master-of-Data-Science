import sys
from typing import List

def solve() -> None:
    """
    Description:
        Основная функция для решения задачи.
        Читает входные данные, находит максимальное число X,
        для которого выполняется условие: sum(min(a[i], X)) >= X * k.

    Args:
        None

    Returns:
        None

    Raises:
        None

    Examples:
        >>> solve()
        # Ввод: 3 4 1 2 3 4
        # Вывод: 2
    """
    data = sys.stdin.read().strip().split()
    if not data:
        return

    k = int(data[0])
    n = int(data[1])
    a = list(map(int, data[2:2 + n]))

    # Максимальное количество советов не может превышать sum(a) // k
    lo = 0
    hi = sum(a) // k
    ans = 0

    # Будем искать по двоичному поиску (методу дихотомии) максимальное число X,
    # для которого выполняется условие: sum(min(a[i], X)) >= X * k.
    while lo <= hi:
        mid = (lo + hi) // 2
        total = 0
        for ai in a:
            total += min(ai, mid)
        if total >= mid * k:
            ans = mid
            lo = mid + 1
        else:
            hi = mid - 1

    sys.stdout.write(str(ans))

if __name__ == '__main__':
    solve()
