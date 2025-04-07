import collections
from typing import List, Tuple

INFINITY = float('inf')

def read_input(file_path: str) -> Tuple[int, int, List[List[int]]]:
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
        Exception: В случае ошибки чтения файла

    Examples:
    ---------------
        >>> read_input("INPUT.TXT")
        (3, 3, [[0, 1, 0], [0, 0, 0], [0, 0, 1]])
    """
    try:
        with open(file_path, "r") as file:
            n, m = map(int, file.readline().split())
            grid = [list(map(int, file.readline().split())) for _ in range(n)]
        return n, m, grid
    except Exception as e:
        print(f"Критическая ошибка при чтении INPUT.TXT: {e}")
        exit(1)

def initialize_distances(n: int, m: int) -> List[List[int]]:
    """
    Description:
    ---------------
        Инициализирует матрицу расстояний.

    Args:
    ---------------
        n: Количество строк
        m: Количество столбцов

    Returns:
    ---------------
        Матрица расстояний, заполненная бесконечностями

    Examples:
    ---------------
        >>> initialize_distances(3, 3)
        [[inf, inf, inf], [inf, inf, inf], [inf, inf, inf]]
    """
    return [[INFINITY] * m for _ in range(n)]

def find_sources(grid: List[List[int]], distances: List[List[int]]) -> collections.deque:
    """
    Description:
    ---------------
        Находит источники (клетки с '1') и инициализирует их расстояния.

    Args:
    ---------------
        grid: Сетка значений
        distances: Матрица расстояний

    Returns:
    ---------------
        Очередь с координатами источников

    Examples:
    ---------------
        >>> find_sources([[0, 1, 0], [0, 0, 0], [0, 0, 1]], initialize_distances(3, 3))
        deque([(0, 1), (2, 2)])
    """
    q = collections.deque()
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if grid[r][c] == 1:
                distances[r][c] = 0
                q.append((r, c))
    return q

def bfs(grid: List[List[int]], distances: List[List[int]], q: collections.deque) -> None:
    """
    Description:
    ---------------
        Выполняет поиск в ширину (BFS) для вычисления кратчайших расстояний.

    Args:
    ---------------
        grid: Сетка значений
        distances: Матрица расстояний
        q: Очередь с координатами источников

    Examples:
    ---------------
        >>> distances = initialize_distances(3, 3)
        >>> q = find_sources([[0, 1, 0], [0, 0, 0], [0, 0, 1]], distances)
        >>> bfs([[0, 1, 0], [0, 0, 0], [0, 0, 1]], distances, q)
        >>> distances
        [[2, 0, 1], [1, 2, 3], [0, 1, 0]]
    """
    # Определяем смещения для соседей (вверх, вниз, влево, вправо)
    dr = [-1, 1, 0, 0]
    dc = [0, 0, -1, 1]

    while q:
        r, c = q.popleft()
        current_dist = distances[r][c]

        # Перебираем 4 возможных направления
        for i in range(4):
            nr, nc = r + dr[i], c + dc[i]

            # Проверяем, что сосед находится в пределах таблицы
            if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]):
                # Если через текущую клетку (r, c) путь до соседа (nr, nc) короче
                # и мы еще не нашли оптимальный путь до него
                if distances[nr][nc] > current_dist + 1:
                    distances[nr][nc] = current_dist + 1
                    q.append((nr, nc))  # Добавляем соседа в очередь для дальнейшей обработки

def write_output(file_path: str, distances: List[List[int]]) -> None:
    """
    Description:
    ---------------
        Записывает результат в файл.

    Args:
    ---------------
        file_path: Путь к файлу для записи результата
        distances: Матрица расстояний

    Raises:
    ---------------
        Exception: В случае ошибки записи файла

    Examples:
    ---------------
        >>> write_output("OUTPUT.TXT", [[2, 0, 1], [1, 2, 3], [0, 1, 0]])
    """
    try:
        with open(file_path, "w") as file:
            for row in distances:
                file.write(" ".join(map(str, row)) + "\n")
    except Exception as e:
        print(f"Критическая ошибка при записи OUTPUT.TXT: {e}")
        exit(1)

def main():
    """
    Description:
    ---------------
        Основная функция для выполнения алгоритма.
    """
    n, m, grid = read_input("INPUT.TXT")
    distances = initialize_distances(n, m)
    q = find_sources(grid, distances)
    bfs(grid, distances, q)
    write_output("OUTPUT.TXT", distances)

if __name__ == "__main__":
    main()