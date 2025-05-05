import sys
from typing import List, Tuple

def read_input(file_path: str) -> Tuple[int, int, List[List[str]]]:
    """
    Description:
    ---------------
        Считывает входные данные из файла.

    Args:
    ---------------
        file_path: Путь к файлу с входными данными

    Returns:
    ---------------
        Кортеж с количеством строк, столбцов и списком значений сетки

    Raises:
    ---------------
        Exception: В случае ошибки чтения файла или невалидного формата данных

    Examples:
    ---------------
        >>> read_input("INPUT.TXT")
        (3, 3, [['.', '*', '.'], ['.', '.', '.'], ['*', '.', '*']])
    """
    try:
        with open(file_path, "r") as file:
            line1 = file.readline().split()
            if len(line1) != 2:
                # Невалидный формат первой строки
                exit(1)
            n_str, m_str = line1
            # Проверка, что N и M действительно числа
            if not n_str.isdigit() or not m_str.isdigit():
                # N или M не являются числами
                exit(1)
            n, m = int(n_str), int(m_str)

            # Проверка ограничений N и M
            if not (1 <= n <= 300 and 1 <= m <= 300):
                # N или M вне допустимого диапазона
                exit(1)

            grid = [list(file.readline().strip()) for _ in range(n)]

            # Проверка фактических размеров прочитанной карты
            if len(grid) != n or any(len(row) != m for row in grid):
                # Несоответствие размеров
                exit(1)

            # Проверка допустимых символов
            allowed_chars = {'.', '*'}
            for r in range(n):
                for c in range(m):
                    if grid[r][c] not in allowed_chars:
                        # Недопустимый символ в карте
                        exit(1)

        return n, m, grid
    except FileNotFoundError:
        # Файл не найден
        exit(1)
    except Exception as e:
        print(f"Error during input processing: {e}", file=sys.stderr)
        exit(1)

def initialize_visited(n: int, m: int) -> List[List[bool]]:
    """
    Description:
    ---------------
        Инициализирует матрицу посещенных клеток.

    Args:
    ---------------
        n: Количество строк
        m: Количество столбцов

    Returns:
    ---------------
        Матрица посещенных клеток, заполненная False

    Examples:
    ---------------
        >>> initialize_visited(3, 3)
        [[False, False, False], [False, False, False], [False, False, False]]
    """
    return [[False] * m for _ in range(n)]

def mark_component(grid: List[List[str]], visited: List[List[bool]], start_r: int, start_c: int) -> None:
    """
    Description:
    ---------------
        Итеративно обходит и маркирует созвездие.

    Args:
    ---------------
        grid: Сетка значений
        visited: Матрица посещенных клеток
        start_r: Начальная строка
        start_c: Начальный столбец

    Examples:
    ---------------
        >>> visited = initialize_visited(3, 3)
        >>> mark_component([['.', '*', '.'], ['.', '.', '.'], ['*', '.', '*']], visited, 0, 1)
        >>> visited
        [[False, True, False], [False, False, False], [True, False, True]]
    """
    # Определяем смещения для соседей (вверх, вниз, влево, вправо)
    dr = [-1, 1, 0, 0]
    dc = [0, 0, -1, 1]

    # Используем стек для итеративного DFS
    stack = [(start_r, start_c)]
    visited[start_r][start_c] = True  # Помечаем стартовую клетку

    while stack:
        curr_r, curr_c = stack.pop()

        # Обходим соседей
        for i in range(4):
            nr, nc = curr_r + dr[i], curr_c + dc[i]

            # Проверка границ, типа клетки ('*') и посещенности
            if (0 <= nr < len(grid) and 0 <= nc < len(grid[0])
                    and not visited[nr][nc]
                    and grid[nr][nc] == '*'):
                visited[nr][nc] = True  # Помечаем как посещенную
                stack.append((nr, nc))  # Добавляем в стек для обхода

def write_output(file_path: str, constellation_count: int) -> None:
    """
    Description:
    ---------------
        Записывает результат в файл.

    Args:
    ---------------
        file_path: Путь к файлу для записи результата
        constellation_count: Количество созвездий

    Raises:
    ---------------
        Exception: В случае ошибки записи файла

    Examples:
    ---------------
        >>> write_output("OUTPUT.TXT", 2)
    """
    try:
        with open(file_path, "w") as file:
            file.write(str(constellation_count) + "\n")
    except Exception as e:
        # Ошибка записи файла
        print(f"Error writing output file: {e}", file=sys.stderr)
        exit(1)

def solve():
    """
    Description:
    ---------------
        Основная функция для выполнения алгоритма.
    """
    n, m, grid = read_input("INPUT.TXT")
    visited = initialize_visited(n, m)
    constellation_count = 0

    # Основной цикл обхода карты
    for r in range(n):
        for c in range(m):
            # Если нашли непосещенную звезду - это новое созвездие
            if grid[r][c] == '*' and not visited[r][c]:
                constellation_count += 1
                mark_component(grid, visited, r, c)  # Запускаем обход для пометки всей компоненты

    write_output("OUTPUT.TXT", constellation_count)

if __name__ == "__main__":
    solve()
