# Импорты стандартных библиотек
import sys
import heapq

def main() -> None:
    """
    Description:
        Основная функция программы. Считывает входные данные, выполняет бинарный поиск
        для нахождения максимального отношения a_i / b_i и выводит результат.

    Args:
        Нет входных параметров.

    Returns:
        None

    Raises:
        Нет исключений.

    Examples:
        Пример использования:
        $ echo "3 2 1 2 3 4 5 6" | python script.py
        1.5000000000
    """
    data = sys.stdin.read().strip().split()
    
    # Если входные данные отсутствуют, завершаем работу
    if not data:
        return
    
    n = int(data[0])
    k = int(data[1])
    pairs = []
    
    # Формируем список пар (a, b)
    for i in range(n):
        a = float(data[2 + 2 * i])
        b = float(data[2 + 2 * i + 1])
        pairs.append((a, b))
    
    lo = 0.0
    hi = 100000.0  # Максимально возможное отношение: a_i / b_i может быть не больше 100000 / 1 = 100000
    eps = 1e-7     # Точность вычислений
    
    # Бинарный поиск по искомому отношению
    while hi - lo > eps:
        mid = (lo + hi) / 2
        # Для каждой пары вычисляем значение a - mid * b
        v = [a - mid * b for a, b in pairs]
        # Выбираем k пар с наибольшими значениями и проверяем их сумму
        largest = heapq.nlargest(k, v)
        if sum(largest) >= 0:
            lo = mid
        else:
            hi = mid
 
    sys.stdout.write("{:.10f}".format(lo))