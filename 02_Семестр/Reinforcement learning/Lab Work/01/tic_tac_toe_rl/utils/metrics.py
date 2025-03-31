#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Модуль для анализа производительности агентов в игре "Крестики-нолики".
Предоставляет функции для сравнения агентов, анализа их стратегий и
визуализации результатов.
"""

# Стандартные библиотеки
from collections import defaultdict
from typing import Dict, List, Any, Optional, Union, Callable, Tuple

# Библиотеки для анализа данных
import numpy as np
import pandas as pd

# Библиотеки для визуализации
import matplotlib.pyplot as plt
import seaborn as sns


def compute_win_percentage(results: List[str]) -> Dict[str, Union[float, int]]:
    """
    Description:
    ---------------
        Вычисление процента побед, ничьих и поражений на основе списка результатов.
    
    Args:
    ---------------
        results: Список результатов игр ('win', 'draw', 'loss')
        
    Returns:
    ---------------
        Словарь с процентами и количеством побед, ничьих и поражений:
        - win_percentage: Процент побед
        - draw_percentage: Процент ничьих
        - loss_percentage: Процент поражений
        - win_count: Количество побед
        - draw_count: Количество ничьих
        - loss_count: Количество поражений
        - total: Общее количество игр
    
    Examples:
    ---------------
        >>> results = ['win', 'win', 'draw', 'loss', 'win']
        >>> stats = compute_win_percentage(results)
        >>> print(f"Побед: {stats['win_percentage']:.1f}%")
        Побед: 60.0%
    """
    total = len(results)
    wins = results.count('win')
    draws = results.count('draw')
    losses = results.count('loss')
    
    return {
        'win_percentage': (wins / total) * 100 if total > 0 else 0,
        'draw_percentage': (draws / total) * 100 if total > 0 else 0,
        'loss_percentage': (losses / total) * 100 if total > 0 else 0,
        'win_count': wins,
        'draw_count': draws,
        'loss_count': losses,
        'total': total
    }


def compare_agents(
    env: Any,
    agents: List[Any],
    agent_names: List[str],
    n_episodes: int = 100
) -> pd.DataFrame:
    """
    Description:
    ---------------
        Сравнение производительности нескольких агентов в одной среде.
    
    Args:
    ---------------
        env: Среда игры в крестики-нолики
        agents: Список агентов для сравнения
        agent_names: Список имен агентов (соответствует списку agents)
        n_episodes: Количество эпизодов для оценки каждого агента
        
    Returns:
    ---------------
        DataFrame с результатами сравнения, содержащий следующие столбцы:
        - Agent: Имя агента
        - Win Rate (%): Процент побед
        - Draw Rate (%): Процент ничьих
        - Loss Rate (%): Процент поражений
        - Avg Reward: Средняя награда за игру
        - Total Games: Общее количество игр
    
    Raises:
    ---------------
        ValueError: Если количество агентов не соответствует количеству имен
    
    Examples:
    ---------------
        >>> from my_environment import TicTacToeEnv
        >>> from my_agents import RandomAgent, MinimaxAgent
        >>> env = TicTacToeEnv(opponent_type='random')
        >>> agents = [RandomAgent(), MinimaxAgent()]
        >>> names = ['Random', 'Minimax']
        >>> results = compare_agents(env, agents, names, n_episodes=100)
        >>> print(results)
    """
    # Проверка соответствия количества агентов и имен
    if len(agents) != len(agent_names):
        raise ValueError(
            "Количество агентов должно соответствовать количеству имен"
        )
    
    # Инициализация словаря для хранения результатов
    results = {
        name: {'wins': 0, 'draws': 0, 'losses': 0, 'avg_reward': 0}
        for name in agent_names
    }
    
    # Оценка каждого агента
    for i, (agent, name) in enumerate(zip(agents, agent_names)):
        total_reward = 0
        
        for episode in range(n_episodes):
            state = env.reset()
            done = False
            episode_reward = 0
            
            # Игровой цикл
            while not done:
                # Определение доступных действий
                flat_state = state.flatten()
                valid_actions = [
                    i for i, val in enumerate(flat_state) if val == 0
                ]
                
                # Выбор действия агентом (без режима обучения)
                action = agent.choose_action(
                    state, valid_actions, training=False
                )
                
                # Выполнение действия в среде
                next_state, reward, done, info = env.step(action)
                
                # Обновление состояния и накопленной награды
                state = next_state
                episode_reward += reward
            
            # Учет результата игры
            if episode_reward > 0:
                results[name]['wins'] += 1
            elif episode_reward == 0:
                results[name]['draws'] += 1
            else:
                results[name]['losses'] += 1
            
            total_reward += episode_reward
        
        # Расчет средней награды
        results[name]['avg_reward'] = total_reward / n_episodes
    
    # Преобразование результатов в DataFrame
    data = []
    for name, stats in results.items():
        total = stats['wins'] + stats['draws'] + stats['losses']
        data.append({
            'Agent': name,
            'Win Rate (%)': (stats['wins'] / total) * 100 if total > 0 else 0,
            'Draw Rate (%)': (stats['draws'] / total) * 100 if total > 0 else 0,
            'Loss Rate (%)': (stats['losses'] / total) * 100 if total > 0 else 0,
            'Avg Reward': stats['avg_reward'],
            'Total Games': total
        })
    
    return pd.DataFrame(data)


def analyze_opponent_types(
    env_creator: Callable[[str], Any],
    agent: Any,
    opponent_types: List[str],
    n_episodes: int = 100
) -> pd.DataFrame:
    """
    Description:
    ---------------
        Анализ производительности агента против различных типов оппонентов.
    
    Args:
    ---------------
        env_creator: Функция для создания среды с указанным типом оппонента
        agent: Обученный агент для тестирования
        opponent_types: Список строковых идентификаторов типов оппонентов
        n_episodes: Количество эпизодов для оценки против каждого оппонента
        
    Returns:
    ---------------
        DataFrame с результатами анализа, содержащий следующие столбцы:
        - Opponent Type: Тип оппонента
        - Win Rate (%): Процент побед агента
        - Draw Rate (%): Процент ничьих
        - Loss Rate (%): Процент поражений агента
        - Avg Reward: Средняя награда за игру
        - Total Games: Общее количество игр
    
    Examples:
    ---------------
        >>> from my_environment import TicTacToeEnv
        >>> from my_agents import QLearningAgent
        >>> def create_env(opp_type):
        ...     return TicTacToeEnv(opponent_type=opp_type)
        >>> agent = QLearningAgent(state_size=9, action_size=9)
        >>> opponent_types = ['random', 'rule_based', 'minimax']
        >>> results = analyze_opponent_types(create_env, agent, opponent_types)
        >>> print(results)
    """
    # Инициализация словаря для хранения результатов
    results = {
        opp_type: {'wins': 0, 'draws': 0, 'losses': 0, 'avg_reward': 0}
        for opp_type in opponent_types
    }
    
    # Анализ против каждого типа оппонента
    for opp_type in opponent_types:
        # Создание среды с текущим типом оппонента
        env = env_creator(opp_type)
        total_reward = 0
        
        # Проведение игр
        for episode in range(n_episodes):
            state = env.reset()
            done = False
            episode_reward = 0
            
            # Игровой цикл
            while not done:
                # Определение доступных действий
                flat_state = state.flatten()
                valid_actions = [
                    i for i, val in enumerate(flat_state) if val == 0
                ]
                
                # Выбор действия агентом
                action = agent.choose_action(
                    state, valid_actions, training=False
                )
                
                # Выполнение действия
                next_state, reward, done, info = env.step(action)
                
                # Обновление состояния и накопленной награды
                state = next_state
                episode_reward += reward
            
            # Учет результата игры
            if episode_reward > 0:
                results[opp_type]['wins'] += 1
            elif episode_reward == 0:
                results[opp_type]['draws'] += 1
            else:
                results[opp_type]['losses'] += 1
            
            total_reward += episode_reward
        
        # Расчет средней награды
        results[opp_type]['avg_reward'] = total_reward / n_episodes
        
        # Закрытие среды
        env.close()
    
    # Преобразование результатов в DataFrame
    data = []
    for opp_type, stats in results.items():
        total = stats['wins'] + stats['draws'] + stats['losses']
        data.append({
            'Opponent Type': opp_type,
            'Win Rate (%)': (stats['wins'] / total) * 100 if total > 0 else 0,
            'Draw Rate (%)': (stats['draws'] / total) * 100 if total > 0 else 0,
            'Loss Rate (%)': (stats['losses'] / total) * 100 if total > 0 else 0,
            'Avg Reward': stats['avg_reward'],
            'Total Games': total
        })
    
    return pd.DataFrame(data)


def analyze_first_moves(agent: Any, board_size: int = 3) -> Optional[np.ndarray]:
    """
    Description:
    ---------------
        Анализ предпочтений агента для первого хода в игре.
    
    Args:
    ---------------
        agent: Обученный агент с Q-таблицей
        board_size: Размер игрового поля (по умолчанию 3x3)
        
    Returns:
    ---------------
        Матрица с Q-значениями для первого хода или None, если
        пустое поле не найдено в Q-таблице агента
    
    Examples:
    ---------------
        >>> from my_agents import QLearningAgent
        >>> agent = QLearningAgent(state_size=9, action_size=9)
        >>> # Предположим, что агент уже обучен
        >>> q_values = analyze_first_moves(agent)
        >>> print(q_values)
    """
    # Создание пустого поля
    empty_board = np.zeros((board_size, board_size), dtype=np.int8)
    
    # Получение ключа для пустого поля
    state_key = str(empty_board.flatten().tolist())
    
    # Проверка наличия состояния в Q-таблице
    if state_key not in agent.q_table:
        print("Пустое поле не найдено в Q-таблице.")
        return None
    
    # Получение Q-значений для первого хода
    q_values = agent.q_table[state_key].reshape(board_size, board_size)
    
    return q_values


def plot_first_move_heatmap(
    agent: Any,
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Description:
    ---------------
        Визуализация предпочтений агента для первого хода в виде тепловой карты.
    
    Args:
    ---------------
        agent: Обученный агент с Q-таблицей
        title: Заголовок графика (если None, используется стандартный)
        save_path: Путь для сохранения графика (если требуется)
    
    Returns:
    ---------------
        None
    
    Examples:
    ---------------
        >>> from my_agents import QLearningAgent
        >>> agent = QLearningAgent(state_size=9, action_size=9)
        >>> # Предположим, что агент уже обучен
        >>> plot_first_move_heatmap(agent, title="Предпочтения для первого хода")
    """
    # Получение матрицы Q-значений
    q_values = analyze_first_moves(agent)
    
    # Проверка наличия данных
    if q_values is None:
        return
    
    # Создание тепловой карты
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        q_values,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        linewidths=.5
    )
    
    # Установка заголовка
    if title:
        plt.title(title)
    else:
        plt.title('Q-значения для первого хода')
    
    # Сохранение графика, если указан путь
    if save_path:
        plt.savefig(save_path)
    
    plt.show()


def learning_curve_analysis(stats: Dict[str, List]) -> Dict[str, Any]:
    """
    Description:
    ---------------
        Анализ кривой обучения агента и расчет различных метрик.
    
    Args:
    ---------------
        stats: Словарь со статистикой обучения, должен содержать ключи:
            - win_rates: список процентов побед на разных этапах
            - draw_rates: список процентов ничьих
            - loss_rates: список процентов поражений
            - evaluation_episodes: список номеров эпизодов оценки
        
    Returns:
    ---------------
        Словарь с результатами анализа:
        - convergence_episode: Эпизод, когда win_rate достигает 90% от максимального
        - max_win_rate: Максимальный достигнутый процент побед
        - max_win_rate_episode: Эпизод, в котором достигнут максимальный процент побед
        - stability: Стандартное отклонение win_rate в последней трети обучения
        - final_win_rate: Финальный процент побед
        - final_draw_rate: Финальный процент ничьих
        - final_loss_rate: Финальный процент поражений
    
    Raises:
    ---------------
        ValueError: Если в stats отсутствуют необходимые ключи
    
    Examples:
    ---------------
        >>> # Предположим, у нас есть статистика обучения
        >>> stats = {
        ...     'win_rates': [30, 45, 60, 75, 80],
        ...     'draw_rates': [40, 35, 30, 20, 15],
        ...     'loss_rates': [30, 20, 10, 5, 5],
        ...     'evaluation_episodes': [100, 200, 300, 400, 500]
        ... }
        >>> analysis = learning_curve_analysis(stats)
        >>> print(f"Максимальный процент побед: {analysis['max_win_rate']}")
        Максимальный процент побед: 80
    """
    # Проверка наличия необходимых ключей
    required_keys = ['win_rates', 'draw_rates', 'loss_rates', 'evaluation_episodes']
    for key in required_keys:
        if key not in stats:
            raise ValueError(f"Статистика должна содержать ключ '{key}'")
    
    # Преобразование данных в массивы NumPy
    win_rates = np.array(stats['win_rates'])
    draw_rates = np.array(stats['draw_rates'])
    loss_rates = np.array(stats['loss_rates'])
    episodes = np.array(stats['evaluation_episodes'])
    
    # Скорость сходимости (эпизод, когда win_rate достигает 90% от максимального)
    max_win_rate = np.max(win_rates)
    convergence_threshold = 0.9 * max_win_rate
    
    try:
        convergence_episode = episodes[
            np.where(win_rates >= convergence_threshold)[0][0]
        ]
    except IndexError:
        # Если порог сходимости не достигнут
        convergence_episode = None
    
    # Максимальный процент побед и соответствующий эпизод
    max_win_rate_episode = episodes[np.argmax(win_rates)]
    
    # Стабильность (стандартное отклонение win_rate в последней трети)
    last_third_idx = len(win_rates) // 3 * 2
    stability = np.std(win_rates[last_third_idx:])
    
    # Финальные показатели
    final_win_rate = win_rates[-1]
    final_draw_rate = draw_rates[-1]
    final_loss_rate = loss_rates[-1]
    
    return {
        'convergence_episode': convergence_episode,
        'max_win_rate': max_win_rate,
        'max_win_rate_episode': max_win_rate_episode,
        'stability': stability,
        'final_win_rate': final_win_rate,
        'final_draw_rate': final_draw_rate,
        'final_loss_rate': final_loss_rate
    }