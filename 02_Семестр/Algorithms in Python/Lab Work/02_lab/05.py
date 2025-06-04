import sys
from typing import List, Tuple

def solve() -> None:
    """
    Description:
        Основная функция для решения задачи.
        Читает входные данные, вычисляет минимальное время t,
        за которое все люди могут собраться в одной точке.

    Args:
        None

    Returns:
        None

    Raises:
        None

    Examples:
        >>> solve()
        # Ввод: 3 1.0 2.0 2.0 3.0 3.0 4.0
        # Вывод: 0.500000
    """
    input_data = sys.stdin.read().split()
    if not input_data:
        return

    n = int(input_data[0])
    people: List[Tuple[float, float]] = []
    idx = 1
    for _ in range(n):
        x = float(input_data[idx])
        v = float(input_data[idx + 1])
        idx += 2
        people.append((x, v))

    def can_meet(t: float) -> bool:
        """
        Description:
            Проверяет, можно ли за время t выбрать точку,
            до которой успеют добраться все люди.

        Args:
            t: Время, за которое все люди должны собраться в одной точке

        Returns:
            True, если можно выбрать такую точку, иначе False

        Raises:
            None

        Examples:
            >>> can_meet(0.5)
            True
        """
        left_bound = -1e18
        right_bound = 1e18
        for x, v in people:
            left_bound = max(left_bound, x - v * t)
            right_bound = min(right_bound, x + v * t)
        return left_bound <= right_bound

    lo = 0.0
    hi = 1e10      # достаточно большое значение для верхней границы времени
    for _ in range(100):  # 100 итераций для достижения необходимой точности
        mid = (lo + hi) / 2.0
        if can_meet(mid):
            hi = mid
        else:
            lo = mid

    # Вывод ответа в виде вещественного числа с 6 знаками после запятой
    print("{:.6f}".format(hi))
