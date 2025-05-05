"""
Программа находит минимальное время прибытия из деревни d в деревню v
с использованием рейсов с расписанием. Используется алгоритм Дейкстры
с учетом временных ограничений на рейсы.

Функциональное назначение:
1. Чтение входных данных из файла INPUT.TXT:
   - N - количество деревень
   - d и v - начальная и конечная деревни
   - R - количество рейсов
   - Описания рейсов (s_node, dep_time, e_node, arr_time)
2. Построение графа рейсов с временными ограничениями
3. Нахождение минимального времени прибытия с помощью алгоритма Дейкстры
4. Запись результата в файл OUTPUT.TXT

Особенности реализации:
- Поддержка до 100 деревень
- Проверка корректности входных данных
- Обработка временных ограничений на рейсы
- Оптимизированная реализация алгоритма Дейкстры
- Подробная обработка ошибок
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
        для нахождения минимального времени прибытия с учетом расписания рейсов.

    Raises:
    ---------------
        SystemExit: В случае ошибок ввода/вывода или некорректных данных
    """
    # --- Чтение входных данных с валидацией ---
    try:
        with open("INPUT.TXT", "r") as f_in:
            # Чтение количества деревень N
            n_line = f_in.readline()
            if not n_line:
                raise ValueError("Файл пуст или отсутствует строка с N.")
            
            try:
                n = int(n_line)
            except ValueError:
                raise ValueError("Некорректный формат числа N.")
            
            if not (1 <= n <= 100):
                raise ValueError(
                    f"Количество деревень N={n} вне допустимого диапазона [1, 100]."
                )

            # Чтение начальной (d) и конечной (v) деревень
            dv_line = f_in.readline()
            if not dv_line:
                raise ValueError("Отсутствует строка с d и v.")
            
            dv_parts = dv_line.split()
            if len(dv_parts) != 2:
                raise ValueError("Строка с d и v должна содержать ровно 2 числа.")
            
            try:
                d, v = map(int, dv_parts)
            except ValueError:
                raise ValueError("Некорректный формат чисел d или v.")
            
            if not (1 <= d <= n and 1 <= v <= n):
                raise ValueError(
                    f"Номера деревень d={d} или v={v} вне допустимого "
                    f"диапазона [1, {n}]."
                )

            # Чтение количества рейсов R
            r_line = f_in.readline()
            if not r_line:
                raise ValueError("Отсутствует строка с количеством рейсов R.")
            
            try:
                r = int(r_line)
            except ValueError:
                raise ValueError("Некорректный формат числа R.")
            
            if r < 0:
                raise ValueError(f"Количество рейсов R={r} не может быть отрицательным.")
            if r > 10000:
                print(
                    f"Предупреждение: Количество рейсов R={r} > 10000, "
                    "но продолжаем обработку."
                )

            # --- Построение графа рейсов ---
            # flights[i] содержит список кортежей (dep_time, dest_node_idx, arr_time)
            # для рейсов из деревни i (0-based индексация)
            flights: List[List[Tuple[int, int, int]]] = [[] for _ in range(n)]
            
            for i in range(r):
                line = f_in.readline()
                if not line:
                    raise ValueError(
                        f"Ожидалось {r} рейсов, но файл закончился на рейсе #{i+1}."
                    )
                
                parts = line.split()
                if len(parts) != 4:
                    raise ValueError(
                        f"Рейс #{i+1}: Ожидалось 4 числа, получено {len(parts)}."
                    )
                
                try:
                    s_node, dep_time, e_node, arr_time = map(int, parts)
                except ValueError:
                    raise ValueError(f"Рейс #{i+1}: Некорректный формат чисел.")

                # Валидация данных рейса
                if not (1 <= s_node <= n and 1 <= e_node <= n):
                    raise ValueError(
                        f"Рейс #{i+1}: Неверные номера деревень ({s_node}, {e_node}). "
                        f"Должны быть от 1 до {n}."
                    )
                
                if not (0 <= dep_time <= 10000 and 0 <= arr_time <= 10000):
                    raise ValueError(
                        f"Рейс #{i+1}: Неверное время ({dep_time}, {arr_time}). "
                        "Должно быть от 0 до 10000."
                    )
                
                if dep_time > arr_time:
                    print(
                        f"Предупреждение: Рейс #{i+1} из {s_node} в {e_node}: "
                        f"время отправления {dep_time} > времени прибытия {arr_time}."
                    )
                    # Продолжаем обработку, несмотря на некорректное время

                # Добавляем рейс (используем 0-based индексацию)
                start_idx = s_node - 1
                end_idx = e_node - 1
                flights[start_idx].append((dep_time, end_idx, arr_time))

    except FileNotFoundError:
        print("Ошибка: Файл INPUT.TXT не найден.")
        sys.exit(1)
    except ValueError as ve:
        print(f"Ошибка в формате входных данных: {ve}")
        sys.exit(1)
    except Exception as e:
        print(f"Непредвиденная ошибка при чтении: {e}")
        sys.exit(1)

    # --- Алгоритм Дейкстры для поиска минимального времени прибытия ---
    start_node = d - 1  # Переводим в 0-based индекс
    target_node = v - 1

    # Обработка случая, когда начальная и конечная деревни совпадают
    if start_node == target_node:
        result = 0
    else:
        min_arrival: List[float] = [INF] * n
        pq: List[Tuple[int, int]] = []  # Приоритетная очередь (min-heap)

        # Инициализация начального состояния
        min_arrival[start_node] = 0
        heapq.heappush(pq, (0, start_node))  # (время прибытия, индекс узла)

        result = -1  # Значение по умолчанию (недостижимо)

        while pq:
            current_time, u = heapq.heappop(pq)

            # Пропускаем, если уже найдено лучшее время для u
            if current_time > min_arrival[u]:
                continue

            # Проверка достижения цели
            if u == target_node:
                result = current_time
                break  # Найден оптимальный путь

            # Обработка исходящих рейсов из текущей деревни u
            for dep_time, neighbor_w, arr_time in flights[u]:
                # Проверяем, успеваем ли на рейс
                if current_time <= dep_time:
                    # Проверяем, улучшает ли этот рейс время прибытия
                    if arr_time < min_arrival[neighbor_w]:
                        min_arrival[neighbor_w] = arr_time
                        heapq.heappush(pq, (arr_time, neighbor_w))

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