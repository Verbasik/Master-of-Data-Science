#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Модуль для обучения и оценки агентов в игре "Крестики-нолики".
Включает функции для тренировки, оценки и визуализации результатов работы агентов.
"""

# Стандартные библиотеки
import os
import time
from typing import Dict, Any, Optional, Union

# Сторонние библиотеки
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def train_agent(
    env: Any,
    agent: Any,
    n_episodes: int = 10000,
    evaluate_every: int = 100,
    n_eval_episodes: int = 100,
    save_dir: str = './models',
    save_every: int = 1000,
    render_training: bool = False
) -> Dict[str, Any]:
    """
    Description:
    ---------------
        Обучение агента через взаимодействие со средой игры "Крестики-нолики".
    
    Args:
    ---------------
        env: Среда игры
        agent: Агент для обучения
        n_episodes: Количество эпизодов обучения
        evaluate_every: Частота оценки производительности
        n_eval_episodes: Количество эпизодов для оценки
        save_dir: Директория для сохранения моделей
        save_every: Частота сохранения модели
        render_training: Флаг отображения процесса обучения
        
    Returns:
    ---------------
        Словарь со статистикой обучения:
        - rewards: Список наград за каждый эпизод
        - win_rates: Список процентов побед на разных этапах обучения
        - draw_rates: Список процентов ничьих на разных этапах обучения
        - loss_rates: Список процентов поражений на разных этапах обучения
        - exploration_rates: Список скоростей исследования на разных этапах
        - evaluation_episodes: Список номеров эпизодов, в которых проводилась оценка
        - training_time: Общее время обучения в секундах
    
    Raises:
    ---------------
        FileNotFoundError: Если невозможно создать директорию для сохранения
    
    Examples:
    ---------------
        >>> from my_environment import TicTacToeEnv
        >>> from my_agent import QLearningAgent
        >>> env = TicTacToeEnv(opponent_type='random')
        >>> agent = QLearningAgent(state_size=9, action_size=9)
        >>> stats = train_agent(env, agent, n_episodes=5000)
        >>> print(f"Финальный процент побед: {stats['win_rates'][-1]:.2f}")
    """
    # Создание директории для сохранения моделей, если она не существует
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Инициализация переменных для отслеживания прогресса
    rewards = []
    win_rates = []
    draw_rates = []
    loss_rates = []
    exploration_rates = []
    evaluation_episodes = []
    
    start_time = time.time()
    
    # Цикл обучения
    for episode in tqdm(range(1, n_episodes + 1), desc="Training"):
        state = env.reset()
        done = False
        total_reward = 0
        
        # Цикл взаимодействия в рамках одного эпизода
        while not done:
            # Определение доступных действий (пустые клетки)
            flat_state = state.flatten()
            valid_actions = [i for i, val in enumerate(flat_state) if val == 0]
            
            # Выбор действия
            action = agent.choose_action(state, valid_actions, training=True)
            
            # Выполнение действия
            next_state, reward, done, info = env.step(action)
            
            # Визуализация, если требуется
            if render_training and episode % evaluate_every == 0:
                env.render()
                time.sleep(0.1)
            
            # Обучение агента
            agent.learn(state, action, reward, next_state, done)
            
            # Обновление состояния и накопленной награды
            state = next_state
            total_reward += reward
        
        # Сохранение результатов эпизода
        rewards.append(total_reward)
        
        # Периодическая оценка производительности
        if episode % evaluate_every == 0 or episode == n_episodes:
            print(f"\nEpisode {episode}/{n_episodes}")
            eval_stats = evaluate_agent(env, agent, n_episodes=n_eval_episodes)
            
            win_rates.append(eval_stats['win_rate'])
            draw_rates.append(eval_stats['draw_rate'])
            loss_rates.append(eval_stats['loss_rate'])
            exploration_rates.append(agent.exploration_rate)
            evaluation_episodes.append(episode)
            
            print(
                f"Win Rate: {eval_stats['win_rate']:.2f}, "
                f"Draw Rate: {eval_stats['draw_rate']:.2f}, "
                f"Loss Rate: {eval_stats['loss_rate']:.2f}"
            )
            print(f"Exploration Rate: {agent.exploration_rate:.4f}")
            print(f"Q-table size: {len(agent.q_table)}")
        
        # Сохранение модели
        if hasattr(agent, 'save') and (
            episode % save_every == 0 or episode == n_episodes
        ):
            agent.save(f"{save_dir}/q_agent_episode_{episode}.pkl")
    
    # Измерение времени обучения
    training_time = time.time() - start_time
    
    # Сохранение финальной модели
    if hasattr(agent, 'save'):
        agent.save(f"{save_dir}/q_agent_final.pkl")
    
    # Формирование статистики обучения
    training_stats = {
        'rewards': rewards,
        'win_rates': win_rates,
        'draw_rates': draw_rates,
        'loss_rates': loss_rates,
        'exploration_rates': exploration_rates,
        'evaluation_episodes': evaluation_episodes,
        'training_time': training_time
    }
    
    return training_stats


def evaluate_agent(
    env: Any,
    agent: Any,
    n_episodes: int = 100,
    render: bool = False
) -> Dict[str, Union[float, int]]:
    """
    Description:
    ---------------
        Оценка производительности агента в среде игры "Крестики-нолики".
    
    Args:
    ---------------
        env: Среда игры
        agent: Агент для оценки
        n_episodes: Количество эпизодов для оценки
        render: Флаг отображения игры
        
    Returns:
    ---------------
        Словарь со статистикой производительности:
        - win_rate: Доля побед
        - draw_rate: Доля ничьих
        - loss_rate: Доля поражений
        - avg_reward: Средняя награда за эпизод
        - wins: Количество побед
        - draws: Количество ничьих
        - losses: Количество поражений
    
    Examples:
    ---------------
        >>> from my_environment import TicTacToeEnv
        >>> from my_agent import QLearningAgent
        >>> env = TicTacToeEnv(opponent_type='random')
        >>> agent = QLearningAgent(state_size=9, action_size=9)
        >>> stats = evaluate_agent(env, agent, n_episodes=1000)
        >>> print(f"Процент побед: {stats['win_rate'] * 100:.1f}%")
    """
    wins = 0
    draws = 0
    losses = 0
    total_reward = 0
    
    for episode in range(n_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # Определение доступных действий
            flat_state = state.flatten()
            valid_actions = [i for i, val in enumerate(flat_state) if val == 0]
            
            # Выбор действия без исследования
            action = agent.choose_action(state, valid_actions, training=False)
            
            # Выполнение действия
            next_state, reward, done, info = env.step(action)
            
            # Визуализация, если требуется
            if render:
                env.render()
                time.sleep(0.5)
            
            # Обновление состояния и накопленной награды
            state = next_state
            episode_reward += reward
        
        # Учет результата игры
        if episode_reward > 0:
            wins += 1
        elif episode_reward == 0:
            draws += 1
        else:
            losses += 1
        
        total_reward += episode_reward
    
    # Расчет показателей
    win_rate = wins / n_episodes
    draw_rate = draws / n_episodes
    loss_rate = losses / n_episodes
    avg_reward = total_reward / n_episodes
    
    return {
        'win_rate': win_rate,
        'draw_rate': draw_rate,
        'loss_rate': loss_rate,
        'avg_reward': avg_reward,
        'wins': wins,
        'draws': draws,
        'losses': losses
    }


def plot_training_stats(
    stats: Dict[str, Any],
    save_path: Optional[str] = None
) -> None:
    """
    Description:
    ---------------
        Визуализация статистики обучения агента.
    
    Args:
    ---------------
        stats: Словарь со статистикой обучения
        save_path: Путь для сохранения графиков (если требуется)
    
    Returns:
    ---------------
        None
    
    Raises:
    ---------------
        ValueError: Если статистика не содержит необходимых ключей
    
    Examples:
    ---------------
        >>> from my_environment import TicTacToeEnv
        >>> from my_agent import QLearningAgent
        >>> env = TicTacToeEnv(opponent_type='random')
        >>> agent = QLearningAgent(state_size=9, action_size=9)
        >>> stats = train_agent(env, agent, n_episodes=5000)
        >>> plot_training_stats(stats, save_path='training_results.png')
    """
    # Проверка наличия необходимых ключей
    required_keys = [
        'rewards', 'win_rates', 'draw_rates', 'loss_rates',
        'exploration_rates', 'evaluation_episodes'
    ]
    
    for key in required_keys:
        if key not in stats:
            raise ValueError(f"Статистика обучения должна содержать ключ '{key}'")
    
    # Создание фигуры с несколькими графиками
    plt.figure(figsize=(15, 10))
    
    # График скользящего среднего наград
    plt.subplot(2, 2, 1)
    plt.plot(
        np.arange(len(stats['rewards'])),
        stats['rewards'],
        alpha=0.3,
        color='blue'
    )
    
    # Добавление скользящего среднего для лучшей визуализации
    window_size = min(100, len(stats['rewards']))
    if window_size > 0:
        smoothed_rewards = np.convolve(
            stats['rewards'],
            np.ones(window_size) / window_size,
            mode='valid'
        )
        plt.plot(
            np.arange(len(smoothed_rewards)) + window_size - 1,
            smoothed_rewards,
            color='blue'
        )
    
    plt.xlabel('Эпизод')
    plt.ylabel('Награда')
    plt.title('Средняя награда за эпизод')
    plt.grid(True)
    
    # График процента побед/ничьих/поражений
    plt.subplot(2, 2, 2)
    plt.plot(
        stats['evaluation_episodes'],
        stats['win_rates'],
        label='Победы',
        color='green'
    )
    plt.plot(
        stats['evaluation_episodes'],
        stats['draw_rates'],
        label='Ничьи',
        color='blue'
    )
    plt.plot(
        stats['evaluation_episodes'],
        stats['loss_rates'],
        label='Поражения',
        color='red'
    )
    plt.xlabel('Эпизод')
    plt.ylabel('Процент')
    plt.title('Динамика результатов игр')
    plt.legend()
    plt.grid(True)
    
    # График скорости исследования
    plt.subplot(2, 2, 3)
    plt.plot(stats['evaluation_episodes'], stats['exploration_rates'])
    plt.xlabel('Эпизод')
    plt.ylabel('Скорость исследования')
    plt.title('Динамика скорости исследования')
    plt.grid(True)
    
    # Суммарная статистика
    plt.subplot(2, 2, 4)
    labels = ['Победы', 'Ничьи', 'Поражения']
    final_stats = [
        stats['win_rates'][-1],
        stats['draw_rates'][-1],
        stats['loss_rates'][-1]
    ]
    colors = ['green', 'blue', 'red']
    plt.bar(labels, final_stats, color=colors)
    plt.ylabel('Процент')
    plt.title('Финальные результаты')
    
    for i, v in enumerate(final_stats):
        plt.text(i, v + 0.01, f'{v:.2f}', ha='center')
    
    plt.tight_layout()
    
    # Сохранение графика, если указан путь
    if save_path:
        plt.savefig(save_path)
    
    plt.show()