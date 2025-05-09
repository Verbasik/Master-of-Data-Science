"""
Программа реализует алгоритм Флойда-Уоршелла для построения транзитивного замыкания графа.
Транзитивное замыкание графа - это матрица достижимости, где reach[i][j] = 1, если существует
путь из вершины i в вершину j, и 0 в противном случае.

Функциональное назначение:
1. Чтение матрицы смежности из файла INPUT.TXT
2. Проверка корректности входных данных (размер матрицы, значения элементов)
3. Построение транзитивного замыкания с помощью алгоритма Флойда-Уоршелла
4. Запись результирующей матрицы достижимости в файл OUTPUT.TXT

Особенности реализации:
- Обработка возможных ошибок ввода/вывода
- Проверка корректности структуры входных данных
- Поддержка графов размером от 1 до 100 вершин
- Детальная валидация входной матрицы смежности
"""

import sys
from typing import List


def solve() -> None:
    """
    Description:
    ---------------
        Основная функция программы, реализующая алгоритм Флойда-Уоршелла
        для построения транзитивного замыкания графа. Читает матрицу смежности
        из файла, строит матрицу достижимости и записывает результат в файл.

    Raises:
    ---------------
        SystemExit: В случае ошибок ввода/вывода или некорректных данных
    """
    # --- Чтение входных данных ---
    try:
        with open("INPUT.TXT", "r") as f_in:
            n = int(f_in.readline())
            # Проверка корректности N
            if not (1 <= n <= 100):
                raise ValueError(
                    f"Количество вершин N={n} вне допустимого диапазона [1, 100]."
                )

            adj_matrix: List[List[int]] = []
            for i in range(n):
                row = list(map(int, f_in.readline().split()))
                # Проверка корректности размера строки и значений
                if len(row) != n:
                    raise ValueError(f"Строка {i+1} матрицы имеет неверную длину.")
                for val in row:
                    if val not in [0, 1]:
                        raise ValueError(
                            f"Недопустимое значение {val} в матрице смежности "
                            "(ожидаются 0 или 1)."
                        )
                # Проверка диагонального элемента (согласно условию)
                if n > 0 and row[i] != 1:
                    print(
                        f"Предупреждение: Гарантия A[{i}][{i}] = 1 не выполнена "
                        f"(A[{i}][{i}] = {row[i]}). Алгоритм продолжит работу."
                    )
                adj_matrix.append(row)

    except FileNotFoundError:
        print("Ошибка: Файл INPUT.TXT не найден.")
        sys.exit(1)
    except ValueError as ve:
        print(f"Ошибка в формате входных данных: {ve}")
        sys.exit(1)
    except Exception as e:
        print(f"Непредвиденная ошибка при чтении входных данных: {e}")
        sys.exit(1)

    # --- Инициализация матрицы достижимости ---
    reach: List[List[int]] = [row[:] for row in adj_matrix]

    # --- Алгоритм Флойда-Уоршелла для транзитивного замыкания ---
    for k in range(n):  # Промежуточная вершина
        for i in range(n):  # Начальная вершина
            for j in range(n):  # Конечная вершина
                # Если из i можно дойти до k И из k можно дойти до j...
                if reach[i][k] == 1 and reach[k][j] == 1:
                    # ...значит, из i можно дойти до j
                    reach[i][j] = 1

    # --- Запись результата в файл ---
    try:
        with open("OUTPUT.TXT", "w") as f_out:
            for i in range(n):
                f_out.write(" ".join(map(str, reach[i])) + "\n")
    except IOError as e:
        print(f"Ошибка при записи в файл OUTPUT.TXT: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Непредвиденная ошибка при записи результата: {e}")
        sys.exit(1)


if __name__ == "__main__":
    solve()