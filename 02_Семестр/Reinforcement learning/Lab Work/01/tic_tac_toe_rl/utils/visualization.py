#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Модуль для визуализации игры "Крестики-нолики" и Q-значений обученного агента.
Предоставляет функции для отображения текущего состояния игры, тепловых карт
Q-значений и анимации игрового процесса.
"""

# Стандартные библиотеки
import os
from typing import Dict, List, Any, Optional, Union, Tuple

# Библиотеки для визуализации
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_q_values_heatmap(
    agent: Any,
    state: np.ndarray,
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Description:
    ---------------
        Визуализация Q-значений для текущего состояния в виде тепловой карты.
    
    Args:
    ---------------
        agent: Обученный агент с Q-таблицей
        state: Текущее состояние игрового поля (матрица 3x3)
        title: Заголовок графика (если None, используется стандартный)
        save_path: Путь для сохранения графика (если требуется)
    
    Returns:
    ---------------
        None
    
    Raises:
    ---------------
        ValueError: Если состояние не соответствует формату 3x3
    
    Examples:
    ---------------
        >>> from my_agents import QLearningAgent
        >>> agent = QLearningAgent(state_size=9, action_size=9)
        >>> # Предположим, что агент уже обучен
        >>> state = np.zeros((3, 3), dtype=np.int8)
        >>> state[0, 0] = 1  # Крестик в верхнем левом углу
        >>> plot_q_values_heatmap(agent, state, title="Q-значения после хода X")
    """
    # Проверка формата состояния
    if state.shape != (3, 3):
        raise ValueError(
            f"Ожидается состояние формата 3x3, получено {state.shape}"
        )
    
    # Получение Q-значений для текущего состояния
    state_key = str(state.flatten().tolist())
    if state_key not in agent.q_table:
        print("Состояние не найдено в Q-таблице.")
        return
    
    q_values = agent.q_table[state_key].copy().reshape(3, 3)
    
    # Маскирование недопустимых действий
    mask = state != 0
    q_values[mask] = np.nan
    
    # Настройка цветовой схемы
    cmap = sns.diverging_palette(10, 133, as_cmap=True)
    
    # Создание тепловой карты
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        q_values,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        linewidths=.5,
        cbar=True
    )
    
    # Добавление заголовка
    if title:
        plt.title(title)
    else:
        plt.title('Q-значения для текущего состояния')
    
    # Сохранение графика, если требуется
    if save_path:
        plt.savefig(save_path)
    
    plt.show()


def visualize_game(
    board: np.ndarray,
    q_agent: Optional[Any] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Description:
    ---------------
        Визуализация текущего состояния игры "Крестики-нолики" и,
        опционально, Q-значений для доступных ходов.
    
    Args:
    ---------------
        board: Текущее состояние игрового поля (матрица 3x3)
        q_agent: Обученный Q-агент (если указан, отображаются Q-значения)
        save_path: Путь для сохранения графика (если требуется)
    
    Returns:
    ---------------
        None
    
    Raises:
    ---------------
        ValueError: Если поле не соответствует формату 3x3
    
    Examples:
    ---------------
        >>> from my_agents import QLearningAgent
        >>> agent = QLearningAgent(state_size=9, action_size=9)
        >>> board = np.array([
        ...     [1, 0, -1],
        ...     [0, 1, 0],
        ...     [0, 0, -1]
        ... ])
        >>> visualize_game(board, agent)
    """
    # Проверка формата игрового поля
    if board.shape != (3, 3):
        raise ValueError(
            f"Ожидается игровое поле формата 3x3, получено {board.shape}"
        )
    
    # Создание фигуры и осей
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Отрисовка сетки
    for i in range(4):
        ax.axhline(i, color='black', linewidth=2)
        ax.axvline(i, color='black', linewidth=2)
    
    # Отрисовка X и O
    for i in range(3):
        for j in range(3):
            if board[i, j] == 1:  # X
                ax.plot(
                    [j + 0.2, j + 0.8],
                    [i + 0.2, i + 0.8],
                    'r-',
                    linewidth=5
                )
                ax.plot(
                    [j + 0.8, j + 0.2],
                    [i + 0.2, i + 0.8],
                    'r-',
                    linewidth=5
                )
            elif board[i, j] == -1:  # O
                circle = plt.Circle(
                    (j + 0.5, i + 0.5),
                    0.3,
                    fill=False,
                    color='blue',
                    linewidth=5
                )
                ax.add_artist(circle)
    
    # Если предоставлен Q-агент, показываем Q-значения для доступных ходов
    if q_agent is not None:
        state_key = str(board.flatten().tolist())
        if state_key in q_agent.q_table:
            q_values = q_agent.q_table[state_key]
            
            for i in range(3):
                for j in range(3):
                    action = i * 3 + j
                    if board[i, j] == 0:  # Пустая клетка
                        # Получение Q-значения
                        q_val = q_values[action]
                        
                        # Размер текста и цвет зависят от Q-значения
                        size = max(10, min(20, abs(q_val) * 10))
                        color = 'green' if q_val > 0 else 'red'
                        
                        # Отображение Q-значения
                        ax.text(
                            j + 0.5,
                            i + 0.5,
                            f'{q_val:.2f}',
                            ha='center',
                            va='center',
                            color=color,
                            size=size,
                            fontweight='bold'
                        )
    
    # Настройка осей
    ax.set_xlim([0, 3])
    ax.set_ylim([0, 3])
    ax.set_xticks([0.5, 1.5, 2.5])
    ax.set_yticks([0.5, 1.5, 2.5])
    ax.set_xticklabels(['0', '1', '2'])
    ax.set_yticklabels(['0', '1', '2'])
    
    # Скрытие осей
    ax.set_axis_off()
    
    # Добавление заголовка
    plt.title('Текущее состояние игры')
    
    # Сохранение графика, если требуется
    if save_path:
        plt.savefig(save_path)
    
    plt.tight_layout()
    plt.show()


def plot_q_distribution(
    agent: Any,
    save_path: Optional[str] = None
) -> None:
    """
    Description:
    ---------------
        Визуализация распределения Q-значений во всей Q-таблице агента.
        Позволяет оценить общее качество обучения.
    
    Args:
    ---------------
        agent: Обученный агент с Q-таблицей
        save_path: Путь для сохранения графика (если требуется)
    
    Returns:
    ---------------
        None
    
    Examples:
    ---------------
        >>> from my_agents import QLearningAgent
        >>> agent = QLearningAgent(state_size=9, action_size=9)
        >>> # Предположим, что агент уже обучен
        >>> plot_q_distribution(agent, save_path="q_distribution.png")
    """
    # Сбор всех Q-значений, исключая -бесконечность
    all_q_values = []
    for state_key, q_values in agent.q_table.items():
        all_q_values.extend([q for q in q_values if q != -np.inf])
    
    # Проверка наличия данных
    if not all_q_values:
        print("Q-таблица пуста или содержит только значения -бесконечность.")
        return
    
    # Создание гистограммы
    plt.figure(figsize=(12, 6))
    sns.histplot(all_q_values, kde=True, bins=50)
    plt.title('Распределение Q-значений')
    plt.xlabel('Q-значение')
    plt.ylabel('Частота')
    plt.grid(True)
    
    # Сохранение графика, если требуется
    if save_path:
        plt.savefig(save_path)
    
    plt.show()


def animate_game(
    env: Any,
    agent: Any,
    max_steps: int = 10,
    save_path: Optional[str] = None
) -> None:
    """
    Description:
    ---------------
        Анимация игры "Крестики-нолики" с использованием обученного агента.
        Визуализирует каждый шаг игры и сохраняет кадры при необходимости.
    
    Args:
    ---------------
        env: Среда игры "Крестики-нолики"
        agent: Обученный агент
        max_steps: Максимальное количество шагов (ограничение для защиты от
                   бесконечной игры)
        save_path: Директория для сохранения кадров анимации
    
    Returns:
    ---------------
        None
    
    Raises:
    ---------------
        FileNotFoundError: Если невозможно создать директорию для сохранения
                          кадров
    
    Examples:
    ---------------
        >>> from my_environment import TicTacToeEnv
        >>> from my_agents import QLearningAgent
        >>> env = TicTacToeEnv(opponent_type='random')
        >>> agent = QLearningAgent(state_size=9, action_size=9)
        >>> # Предположим, что агент уже обучен
        >>> animate_game(env, agent, save_path="game_frames")
    """
    state = env.reset()
    done = False
    step = 0
    
    # Создание директории для сохранения кадров, если требуется
    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Игровой цикл
    while not done and step < max_steps:
        # Вывод информации о текущем шаге
        print(f"Шаг {step + 1}")
        env.render()
        
        # Сохранение кадра, если указан путь
        if save_path:
            visualize_game(
                state,
                agent,
                f"{save_path}/step_{step}.png"
            )
        
        # Определение доступных действий
        flat_state = state.flatten()
        valid_actions = [i for i, val in enumerate(flat_state) if val == 0]
        
        # Выбор действия агентом
        action = agent.choose_action(state, valid_actions, training=False)
        
        # Выполнение действия в среде
        state, reward, done, info = env.step(action)
        step += 1
    
    # Визуализация финального состояния
    print(f"Финальное состояние (шаг {step}):")
    env.render()
    
    # Сохранение финального кадра, если указан путь
    if save_path:
        visualize_game(
            state,
            agent,
            f"{save_path}/step_{step}_final.png"
        )
    
    # Вывод результата игры
    if info.get('message'):
        print(f"Результат: {info['message']}")