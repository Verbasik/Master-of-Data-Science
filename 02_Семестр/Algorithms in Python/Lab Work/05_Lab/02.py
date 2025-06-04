# -*- coding: utf-8 -*-
import sys
from typing import List, Tuple

# Константы для размера сетки
N_ROWS = 8
N_COLS = 8

def solve() -> None:
    """
    Основная функция для решения задачи о минимальном количестве строителей.
    Считает количество связных компонент на сетке 8x8, где связность
    определяется правильным шахматным чередованием цветов ('W' и 'B')
    соседних плиток. Использует итеративный поиск в глубину (DFS).
    """
    # --- 1. Чтение входных данных ---
    grid: List[str] = []
    try:
        with open('INPUT.TXT', 'r', encoding='utf-8') as f:
            for i in range(N_ROWS):
                line = f.readline()
                # Проверка на недостаток строк
                if not line:
                    print(f"Ошибка: Неожиданный конец файла. Ожидалось {N_ROWS} строк, прочитано {i}.", file=sys.stderr)
                    try:
                        open('OUTPUT.TXT', 'w').close()  # Создать пустой файл при ошибке
                    except:
                        pass
                    return
                line = line.strip()
                # Проверка длины строки
                if len(line) != N_COLS:
                    print(f"Ошибка: Строка {i+1} имеет длину {len(line)}, ожидалось {N_COLS}.", file=sys.stderr)
                    try:
                        open('OUTPUT.TXT', 'w').close()
                    except:
                        pass
                    return
                # Проверка допустимых символов
                valid_chars = 'WB'
                if not all(c in valid_chars for c in line):
                    invalid_chars = ''.join(sorted(list(set(c for c in line if c not in valid_chars))))
                    print(f"Ошибка: Строка {i+1} содержит недопустимые символы '{invalid_chars}' (разрешены только 'W' и 'B').", file=sys.stderr)
                    try:
                        open('OUTPUT.TXT', 'w').close()
                    except:
                        pass
                    return
                grid.append(line)
        # Проверка, что прочитано ровно N_ROWS строк (на случай лишних строк)
        if len(grid) != N_ROWS:
            print(f"Ошибка: Прочитано {len(grid)} строк, ожидалось {N_ROWS}.", file=sys.stderr)
            try:
                open('OUTPUT.TXT', 'w').close()
            except:
                pass
            return

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

    # --- 2. Инициализация ---
    visited: List[List[bool]] = [[False for _ in range(N_COLS)] for _ in range(N_ROWS)]
    component_count = 0

    # --- 3. Реализация итеративного DFS ---
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
        return 0 <= r < N_ROWS and 0 <= c < N_COLS

    def dfs_iterative(start_r: int, start_c: int) -> None:
        """
        Итеративный DFS для обхода одной связной компоненты.

        Args:
        ---------------
            start_r: Начальная координата строки
            start_c: Начальная координата столбца
        """
        if visited[start_r][start_c]:  # Доп. проверка, хотя не должна срабатывать
            return

        stack: List[Tuple[int, int]] = [(start_r, start_c)]
        visited[start_r][start_c] = True

        while stack:
            curr_r, curr_c = stack.pop()
            current_color = grid[curr_r][curr_c]

            # Направления смещения к соседям
            directions: List[Tuple[int, int]] = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Право, Лево, Низ, Верх

            for dr, dc in directions:
                nr, nc = curr_r + dr, curr_c + dc

                # Условия для добавления соседа в стек:
                # 1. В пределах сетки
                # 2. Не посещен
                # 3. Цвет отличается (правило шахматной доски)
                if is_valid(nr, nc) and \
                   not visited[nr][nc] and \
                   grid[nr][nc] != current_color:
                    visited[nr][nc] = True
                    stack.append((nr, nc))

    # --- 4. Основной цикл обхода сетки ---
    for i in range(N_ROWS):
        for j in range(N_COLS):
            # Если клетка не посещена, начинаем обход новой компоненты
            if not visited[i][j]:
                component_count += 1
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
