# -*- coding: utf-8 -*-
import sys
from typing import List, Tuple

# Увеличиваем лимит глубины рекурсии для предотвращения RecursionError
# на больших или сложных сетках в пределах N, M <= 100.
# Максимальная глубина может быть N * M = 100 * 100 = 10000.
# Устанавливаем с запасом.
try:
    # Устанавливаем лимит рекурсии (например, 10500)
    sys.setrecursionlimit(10500)
except Exception as e:
    # В некоторых средах изменение лимита может быть ограничено
    # Выводим предупреждение в stderr, чтобы не мешать выводу в OUTPUT.TXT
    print(f"Предупреждение: Не удалось установить лимит рекурсии: {e}. "
          "Возможны проблемы на больших входах.", file=sys.stderr)

def solve() -> None:
    """
    Основная функция для решения задачи поиска связных компонент.
    Использует поиск в глубину (DFS).
    """
    # --- 1. Чтение входных данных ---
    try:
        with open('INPUT.TXT', 'r', encoding='utf-8') as f:
            # Читаем размеры сетки N (строки) и M (столбцы)
            line1 = f.readline()
            if not line1:
                print("Ошибка: Файл INPUT.TXT пуст или первая строка отсутствует.", file=sys.stderr)
                # В случае ошибки создаем пустой OUTPUT.TXT или с 0, чтобы система тестирования не зависла
                try:
                    open('OUTPUT.TXT', 'w').close()
                except:
                    pass
                return
            try:
                n, m = map(int, line1.split())
            except ValueError:
                print("Ошибка: Некорректный формат первой строки в INPUT.TXT (ожидаются два целых числа, разделенных пробелом).", file=sys.stderr)
                try:
                    open('OUTPUT.TXT', 'w').close()
                except:
                    pass
                return

            # Читаем саму сетку
            grid = [f.readline().strip() for _ in range(n)]

    except FileNotFoundError:
        print("Ошибка: Файл INPUT.TXT не найден.", file=sys.stderr)
        try:
            open('OUTPUT.TXT', 'w').close()
        except:
            pass
        return
    except Exception as e:
        print(f"Ошибка при чтении файла INPUT.TXT: {e}", file=sys.stderr)
        try:
            open('OUTPUT.TXT', 'w').close()
        except:
            pass
        return

    # --- Валидация входных данных ---
    validation_error = False
    if not (1 <= n <= 100 and 1 <= m <= 100):
        print(f"Ошибка: Размеры сетки N={n}, M={m} выходят за пределы ограничений [1, 100].", file=sys.stderr)
        validation_error = True
    elif len(grid) != n:
        print(f"Ошибка: Ожидалось {n} строк сетки, но прочитано {len(grid)}.", file=sys.stderr)
        validation_error = True
    else:
        for i, row in enumerate(grid):
            if len(row) != m:
                print(f"Ошибка: Строка {i+1} имеет длину {len(row)}, ожидалось {m}.", file=sys.stderr)
                validation_error = True
                break
            if not all(c in '#.' for c in row):
                print(f"Ошибка: Строка {i+1} содержит недопустимые символы (разрешены только '#' и '.').", file=sys.stderr)
                validation_error = True
                break
    if validation_error:
        try:
            open('OUTPUT.TXT', 'w').close()
        except:
            pass
        return

    # --- 2. Инициализация ---
    # Матрица для отслеживания посещенных клеток '#'
    visited = [[False for _ in range(m)] for _ in range(n)]
    # Счетчик связных компонент (кусков бумаги)
    component_count = 0

    # --- 3. Реализация DFS ---
    def is_valid(r: int, c: int) -> bool:
        """
        Проверяет, находятся ли координаты (r, c) в пределах сетки.

        Args:
        ---------------
            r: Координата строки
            c: Координата столбца

        Returns:
        ---------------
            True, если координаты в пределах сетки, иначе False
        """
        return 0 <= r < n and 0 <= c < m

    # Используем стек для итеративного DFS, чтобы избежать проблем с лимитом рекурсии
    # Это более надежный подход для соревновательного программирования
    def dfs_iterative(start_r: int, start_c: int) -> None:
        """
        Итеративная функция поиска в глубину с использованием стека.

        Args:
        ---------------
            start_r: Начальная координата строки
            start_c: Начальная координата столбца
        """
        if not is_valid(start_r, start_c) or visited[start_r][start_c] or grid[start_r][start_c] == '.':
            # Не начинаем обход, если стартовая точка невалидна
            return

        stack = [(start_r, start_c)]
        # Помечаем стартовую точку
        visited[start_r][start_c] = True

        while stack:
            r, c = stack.pop()

            # Проверяем 4-х соседей
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:  # Вправо, Влево, Вниз, Вверх
                nr, nc = r + dr, c + dc

                # Если сосед валиден, является '#' и еще не посещен
                if is_valid(nr, nc) and not visited[nr][nc] and grid[nr][nc] == '#':
                    visited[nr][nc] = True
                    stack.append((nr, nc))  # Добавляем в стек для дальнейшего обхода

    # --- 4. Основной цикл обхода сетки ---
    # Итерируем по каждой клетке сетки
    for i in range(n):
        for j in range(m):
            # Если находим клетку бумаги ('#'), которая еще не была посещена,
            # значит, мы обнаружили начало новой связной компоненты.
            if grid[i][j] == '#' and not visited[i][j]:
                # Увеличиваем счетчик компонент
                component_count += 1
                # Запускаем итеративный DFS из этой клетки, чтобы найти и пометить
                # все клетки, принадлежащие этой компоненте.
                dfs_iterative(i, j)

    # --- 5. Вывод результата ---
    try:
        with open('OUTPUT.TXT', 'w', encoding='utf-8') as f:
            f.write(str(component_count))
    except IOError:
        print("Ошибка: Не удалось записать результат в файл OUTPUT.TXT.", file=sys.stderr)
    except Exception as e:
        print(f"Ошибка при записи в файл OUTPUT.TXT: {e}", file=sys.stderr)

# Запуск основной функции решения
if __name__ == "__main__":
    solve()
