"""
Программа находит максимальный кратчайший путь между всеми парами вершин
в неориентированном графе с неотрицательными весами, используя алгоритм Флойда-Уоршелла.

Функциональное назначение:
1. Чтение входных данных из файла INPUT.TXT:
   - N - количество вершин (зданий)
   - M - количество ребер (дорог)
   - Описания дорог (s, e, l) - номера зданий и длина дороги между ними
2. Построение матрицы кратчайших расстояний между всеми парами вершин
3. Нахождение максимального значения в матрице расстояний
4. Запись результата в файл OUTPUT.TXT

Особенности реализации:
- Поддержка графов размером от 1 до 100 вершин
- Проверка корректности входных данных
- Обработка ошибок ввода/вывода
- Оптимизация для неориентированных графов
"""

import sys
import math
from typing import List


INF = math.inf


def solve() -> None:
    """
    Description:
    ---------------
        Основная функция программы, реализующая алгоритм Флойда-Уоршелла
        для нахождения максимального кратчайшего пути в графе.

    Raises:
    ---------------
        SystemExit: В случае ошибок ввода/вывода или некорректных данных
    """
    # --- Чтение входных данных ---
    try:
        with open("INPUT.TXT", "r") as f_in:
            line1 = f_in.readline().split()
            if len(line1) != 2:
                raise ValueError(
                    "Первая строка должна содержать ровно два числа: N и M."
                )
            n, m = map(int, line1)

            # Проверка корректности N и M
            if not (1 <= n <= 100):
                raise ValueError(
                    f"Количество зданий N={n} вне допустимого диапазона [1, 100]."
                )
            if not (0 <= m <= 10000):
                print(
                    f"Предупреждение: Количество дорог M={m} кажется большим, "
                    "но продолжаем."
                )

            # Инициализация матрицы расстояний
            dist: List[List[float]] = [[INF] * n for _ in range(n)]
            for i in range(n):
                dist[i][i] = 0

            # Считываем M дорог
            for line_num in range(m):
                line = f_in.readline()
                if not line and line_num < m:
                    raise ValueError(
                        f"Ожидалось {m} строк с ребрами, но файл закончился "
                        f"на строке {line_num+1}."
                    )
                parts = line.split()
                if len(parts) != 3:
                    raise ValueError(
                        f"Строка {line_num+2}: Ожидалось 3 числа (s, e, l), "
                        f"получено {len(parts)}."
                    )

                try:
                    s, e, l = map(int, parts)
                except ValueError:
                    raise ValueError(
                        f"Строка {line_num+2}: Некорректный формат чисел "
                        "в описании дороги."
                    )

                # Проверка корректности номеров зданий и длины дороги
                if not (1 <= s <= n and 1 <= e <= n):
                    raise ValueError(
                        f"Строка {line_num+2}: Неверные номера зданий ({s}, {e}). "
                        f"Должны быть от 1 до {n}."
                    )
                if not (0 <= l <= 100):
                    raise ValueError(
                        f"Строка {line_num+2}: Неверная длина дороги: {l}. "
                        "Должна быть от 0 до 100."
                    )

                # Игнорируем петли
                if s == e:
                    continue

                u, v = s - 1, e - 1
                dist[u][v] = min(dist[u][v], l)
                dist[v][u] = min(dist[v][u], l)

    except FileNotFoundError:
        print("Ошибка: Файл INPUT.TXT не найден.")
        sys.exit(1)
    except ValueError as ve:
        print(f"Ошибка в формате входных данных: {ve}")
        sys.exit(1)
    except Exception as e:
        print(f"Непредвиденная ошибка при чтении входных данных: {e}")
        sys.exit(1)

    # --- Алгоритм Флойда-Уоршелла ---
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] != INF and dist[k][j] != INF:
                    new_dist = dist[i][k] + dist[k][j]
                    if new_dist < dist[i][j]:
                        dist[i][j] = new_dist

    # --- Поиск максимального кратчайшего пути ---
    max_shortest_path = 0
    for i in range(n):
        # Оптимизация для неориентированного графа:
        # проверяем только пары с j > i
        for j in range(i + 1, n):
            if dist[i][j] != INF and dist[i][j] > max_shortest_path:
                max_shortest_path = dist[i][j]

    # --- Запись результата в файл ---
    try:
        with open("OUTPUT.TXT", "w") as f_out:
            f_out.write(str(max_shortest_path) + "\n")
    except IOError as e:
        print(f"Ошибка при записи в файл OUTPUT.TXT: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Непредвиденная ошибка при записи результата: {e}")
        sys.exit(1)


if __name__ == "__main__":
    solve()