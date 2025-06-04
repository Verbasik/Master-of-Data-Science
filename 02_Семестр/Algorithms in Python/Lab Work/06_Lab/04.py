"""
Программа находит путь минимальной стоимости из города 1 в город N,
где стоимость перехода между городами равна стоимости бензина в городе отправления.
Используется алгоритм Дейкстры для нахождения кратчайшего пути.

Функциональное назначение:
1. Чтение входных данных из файла INPUT.TXT:
   - N - количество городов
   - Стоимости бензина в каждом городе
   - M - количество дорог
   - Описания дорог между городами
2. Построение графа (списка смежности)
3. Нахождение пути минимальной стоимости с помощью алгоритма Дейкстры
4. Запись результата в файл OUTPUT.TXT

Особенности реализации:
- Поддержка до 100 городов
- Проверка корректности входных данных
- Оптимизированная обработка ошибок
- Эффективное использование памяти
- Специальная обработка граничных случаев
"""

import sys
import heapq
import math
from typing import List, Tuple


# Константа для представления бесконечности
INF = math.inf


def solve() -> None:
    """
    Description:
    ---------------
        Основная функция программы, реализующая алгоритм Дейкстры
        для нахождения пути минимальной стоимости между городами.

    Raises:
    ---------------
        SystemExit: В случае ошибок ввода/вывода или некорректных данных
    """
    # --- Чтение входных данных с улучшенной обработкой ошибок ---
    try:
        with open("INPUT.TXT", "r") as f_in:
            # Чтение N
            n_line = f_in.readline()
            if not n_line:
                raise ValueError("Файл пуст или отсутствует первая строка с N.")
            try:
                n = int(n_line)
            except ValueError:
                raise ValueError("Некорректный формат числа N в первой строке.")
            
            if not (1 <= n <= 100):
                raise ValueError(
                    f"Количество городов N={n} вне допустимого диапазона [1, 100]."
                )

            # Чтение стоимостей бензина
            costs_line = f_in.readline()
            if not costs_line:
                raise ValueError("Отсутствует строка со стоимостями бензина.")
            
            costs_parts = costs_line.split()
            if len(costs_parts) != n:
                raise ValueError(
                    f"Ожидалось {n} стоимостей бензина, получено {len(costs_parts)}."
                )
            
            try:
                costs = list(map(int, costs_parts))
            except ValueError:
                raise ValueError(
                    "Некорректный формат стоимостей бензина во второй строке."
                )
            
            for i, cost in enumerate(costs):
                if not (0 <= cost <= 100):
                    raise ValueError(
                        f"Стоимость бензина в городе {i+1} ({cost}) "
                        "вне допустимого диапазона [0, 100]."
                    )

            # Чтение количества дорог M
            m = 0
            if n > 1:  # M читаем только если есть хотя бы 2 города
                m_line = f_in.readline()
                if not m_line:
                    raise ValueError("Отсутствует строка с количеством дорог M.")
                
                try:
                    m = int(m_line)
                except ValueError:
                    raise ValueError("Некорректный формат числа M в третьей строке.")
                
                if not (0 <= m <= 10000):
                    print(
                        f"Предупреждение: Количество дорог M={m} выходит за "
                        "разумные пределы, но продолжаем."
                    )

            # --- Построение графа (список смежности) ---
            adj: List[List[int]] = [[] for _ in range(n)]
            for line_num in range(m):
                line = f_in.readline()
                if not line:
                    raise ValueError(
                        f"Ожидалось {m} строк с ребрами, но файл закончился "
                        f"раньше (строка {line_num+1})."
                    )
                
                parts = line.split()
                if len(parts) != 2:
                    raise ValueError(
                        f"Строка ребра #{line_num+1}: Ожидалось 2 числа, "
                        f"получено {len(parts)}."
                    )

                try:
                    u_raw, v_raw = map(int, parts)
                except ValueError:
                    raise ValueError(
                        f"Строка ребра #{line_num+1}: Некорректный формат "
                        "номеров городов."
                    )

                if not (1 <= u_raw <= n and 1 <= v_raw <= n):
                    raise ValueError(
                        f"Строка ребра #{line_num+1}: Неверные номера городов "
                        f"({u_raw}, {v_raw}). Должны быть от 1 до {n}."
                    )
                
                if u_raw == v_raw:
                    continue  # Игнорируем петли

                u, v = u_raw - 1, v_raw - 1
                adj[u].append(v)
                adj[v].append(u)

    except FileNotFoundError:
        print("Ошибка: Файл INPUT.TXT не найден.")
        sys.exit(1)
    except ValueError as ve:
        print(f"Ошибка в формате входных данных: {ve}")
        sys.exit(1)
    except Exception as e:
        print(f"Непредвиденная ошибка при чтении входных данных: {e}")
        sys.exit(1)

    # --- Алгоритм Дейкстры ---
    min_cost: List[float] = [INF] * n
    pq: List[Tuple[float, int]] = []  # Приоритетная очередь (min-heap)

    start_node = 0
    target_node = n - 1

    result = -1  # Изначально считаем, что добраться невозможно

    # Обработка граничных случаев
    if n == 1:  # Если только один город
        result = 0
    elif n > 1:  # Запускаем Дейкстру только если есть хотя бы 2 города
        min_cost[start_node] = 0
        heapq.heappush(pq, (0, start_node))

        while pq:
            current_total_cost, u = heapq.heappop(pq)

            # Пропускаем, если нашли лучший путь ранее
            if current_total_cost > min_cost[u]:
                continue

            # Проверка на достижение цели
            if u == target_node:
                result = int(current_total_cost)  # Стоимость всегда целая
                break  # Найден оптимальный путь

            # Стоимость бензина для выезда из текущего города u
            cost_to_leave_u = costs[u]

            # Релаксация ребер (переходы к соседям)
            for v in adj[u]:
                new_total_cost = current_total_cost + cost_to_leave_u
                if new_total_cost < min_cost[v]:
                    min_cost[v] = new_total_cost
                    heapq.heappush(pq, (new_total_cost, v))

    # --- Запись результата ---
    try:
        with open("OUTPUT.TXT", "w") as f_out:
            f_out.write(str(result) + "\n")
    except IOError as e:
        print(f"Ошибка при записи в файл OUTPUT.TXT: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Непредвиденная ошибка при записи результата: {e}")
        sys.exit(1)


if __name__ == "__main__":
    solve()