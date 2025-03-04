import sys

def solve() -> None:
    """
    Description:
        Основная функция для решения задачи.
        Читает входные данные, вычисляет максимальное количество гамбургеров,
        которые можно приготовить, не превышая бюджет r.

    Args:
        None

    Returns:
        None

    Raises:
        None

    Examples:
        >>> solve()
        # Ввод: BSSC 1 2 3 10 20 30 100
        # Вывод: 3
    """
    input_data = sys.stdin.read().split()
    if not input_data:
        return

    # Чтение входных данных
    recipe = input_data[0]
    n_b, n_s, n_c = map(int, input_data[1:4])
    p_b, p_s, p_c = map(int, input_data[4:7])
    r = int(input_data[7])

    # Подсчитаем требуемое количество ингредиентов для одного гамбургера
    req_b = recipe.count('B')
    req_s = recipe.count('S')
    req_c = recipe.count('C')

    def cost(x: int) -> int:
        """
        Description:
            Вычисляет суммарную стоимость докупки ингредиентов для изготовления x гамбургеров.

        Args:
            x: Количество гамбургеров

        Returns:
            Суммарная стоимость докупки ингредиентов

        Raises:
            None

        Examples:
            >>> cost(3)
            60
        """
        # Дополнительные ингредиенты, которые нужно докупить для x гамбургеров
        buy_b = max(0, req_b * x - n_b)
        buy_s = max(0, req_s * x - n_s)
        buy_c = max(0, req_c * x - n_c)
        return buy_b * p_b + buy_s * p_s + buy_c * p_c

    # Применяем метод дихотомии (бинарного поиска) для нахождения максимального x,
    # такого что стоимость докупки не превышает r.
    lo = 0
    hi = 10**13  # достаточно большое значение для верхней границы
    ans = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if cost(mid) <= r:
            ans = mid
            lo = mid + 1
        else:
            hi = mid - 1

    sys.stdout.write(str(ans))

if __name__ == '__main__':
    solve()
