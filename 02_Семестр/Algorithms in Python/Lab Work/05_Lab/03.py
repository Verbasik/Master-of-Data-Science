# -*- coding: utf-8 -*-
import sys
from collections import deque
from typing import List, Dict, Tuple

# Имена файлов ввода/вывода
input_filename = "INPUT.TXT"
output_filename = "OUTPUT.TXT"

def solve() -> None:
    """
    Решает задачу восстановления карты империи после распада.
    Использует BFS из нескольких источников (столиц) для определения
    принадлежности городов к государствам. Читает из INPUT.TXT, пишет в OUTPUT.TXT.
    """
    capitals_input_order: List[int] = []  # Сохраним исходный порядок столиц для вывода
    capitals: List[int] = []  # Список валидных столиц
    n = 0
    m = 0
    k = 0
    adj: List[List[int]] = []

    try:
        # Чтение входных данных из файла
        with open(input_filename, 'r') as f_in:
            # Чтение N, M и базовая валидация
            line1 = f_in.readline()
            if not line1:
                raise EOFError("Файл пуст или не содержит первой строки (N, M)")
            n_str, m_str = line1.split()
            n = int(n_str)
            m = int(m_str)
            if not (1 <= n <= 1000):
                raise ValueError(f"N={n} вне [1, 1000]")
            if not (0 <= m <= 10**5):
                raise ValueError(f"M={m} вне [0, 10^5]")

            adj = [[] for _ in range(n + 1)]
            for i in range(m):
                line = f_in.readline()
                if not line:
                    raise EOFError(f"Ожидалось ребро {i+1}/{m}, получен EOF")
                u_str, v_str = line.split()
                u = int(u_str)
                v = int(v_str)
                # Валидация ребер
                if not (1 <= u <= n and 1 <= v <= n):
                    continue  # Игнорируем некорректное ребро
                adj[u].append(v)
                adj[v].append(u)

            # Чтение K и валидация
            line_k = f_in.readline()
            if not line_k:
                raise EOFError("Ожидалось K, получен EOF")
            k = int(line_k)
            if not (1 <= k <= n):
                raise ValueError(f"K={k} вне [1, N={n}]")

            # Чтение столиц и валидация
            line_capitals = f_in.readline()
            if not line_capitals:
                raise EOFError("Ожидался список столиц, получен EOF")
            capitals_input_order = list(map(int, line_capitals.split()))

            if len(capitals_input_order) != k:
                raise ValueError(f"Ожидалось K={k} столиц, получено {len(capitals_input_order)}")

            seen_capitals = set()
            for c in capitals_input_order:
                if not (1 <= c <= n):
                    raise ValueError(f"Некорректный номер столицы: {c}")
                if c in seen_capitals:
                    raise ValueError(f"Столица {c} указана более одного раза")
                capitals.append(c)  # Добавляем только валидные и уникальные
                seen_capitals.add(c)

    except FileNotFoundError:
        # Создаем пустой файл при ошибке
        with open(output_filename, "w") as f_out:
            pass
        return
    except (ValueError, EOFError, IndexError) as e:
        # Создаем пустой файл при ошибке
        with open(output_filename, "w") as f_out:
            pass
        return
    except Exception as e:
        # Создаем пустой файл при ошибке
        with open(output_filename, "w") as f_out:
            pass
        return

    # --- Основная логика BFS ---
    owner: List[int] = [0] * (n + 1)  # owner[i] = c, если город i принадлежит столице c
    q: deque = deque()

    # Начальное состояние BFS: добавляем все (валидные) столицы
    for capital in capitals:
        owner[capital] = capital
        q.append(capital)

    # Запускаем BFS
    processed_nodes_count = 0
    while q:
        u = q.popleft()
        processed_nodes_count += 1

        for v in adj[u]:
            if 1 <= v <= n and owner[v] == 0:
                owner[v] = owner[u]  # Присваиваем ту же столицу
                q.append(v)

    # Формирование результата
    states: Dict[int, List[int]] = {c: [] for c in capitals}
    for i in range(1, n + 1):
        if owner[i] != 0:  # Если город был достигнут
            states[owner[i]].append(i)

    # Запись результата в файл
    try:
        with open(output_filename, "w") as f_out:
            output_lines: List[str] = []
            for capital in capitals_input_order:
                if capital in states:
                    result_list = states[capital]
                    output_lines.append(str(len(result_list)))
                    output_lines.append(" ".join(map(str, sorted(result_list))))
                else:
                    output_lines.append("0")
                    output_lines.append("")

            f_out.write("\n".join(output_lines) + "\n")
    except IOError:
        pass
    except Exception:
        pass

# Вызов основной функции решения
solve()
