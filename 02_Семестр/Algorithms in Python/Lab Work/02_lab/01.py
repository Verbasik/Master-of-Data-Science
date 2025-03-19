import sys
import math
from typing import List

def solve() -> None:
    """
    Description:
        Решает уравнение mid * mid + sqrt(mid) = c с использованием метода бинарного поиска.

    Args:
        None

    Returns:
        None

    Raises:
        None

    Examples:
        >>> solve()
        Ввод: 2
        Вывод: 1.0
    """
    input_data: List[str] = sys.stdin.read().strip().split()
    if not input_data:
        return

    c: float = float(input_data[0])

    lo: float = 0.0
    hi: float = c

    for _ in range(100):
        mid: float = (lo + hi) / 2
        if mid * mid + math.sqrt(mid) > c:
            hi = mid
        else:
            lo = mid

    print(lo)

if __name__ == '__main__':
    solve()
