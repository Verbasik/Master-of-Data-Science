from typing import List

def solve_optimized() -> None:
    """
    Description:
    ---------------
        Находит максимальную стоимость подмассива, суммарный вес которого не превышает заданное значение s.

    Args:
    ---------------
        None (ввод осуществляется через стандартный ввод)

    Returns:
    ---------------
        None (вывод осуществляется через стандартный вывод)

    Examples:
    ---------------
        >>> solve_optimized()
        Ввод:
        5 10
        2 3 4 5 6
        3 4 5 6 7
        Вывод:
        15
    """
    n, s = map(int, input().split())
    weights = list(map(int, input().split()))
    costs = list(map(int, input().split()))

    max_cost = 0
    current_weight_sum = 0
    current_cost_sum = 0
    left = 0

    # Проходим по массиву справа налево
    for right in range(n):
        current_weight_sum += weights[right]
        current_cost_sum += costs[right]

        # Поддерживаем текущую сумму весов не больше s
        while current_weight_sum > s:
            current_weight_sum -= weights[left]
            current_cost_sum -= costs[left]
            left += 1

        # Обновляем максимальную стоимость
        max_cost = max(max_cost, current_cost_sum)

    print(max_cost)

if __name__ == "__main__":
    solve_optimized()
