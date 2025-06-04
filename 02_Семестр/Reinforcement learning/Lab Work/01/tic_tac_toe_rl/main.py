#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Основной модуль программы для обучения и использования агентов 
в игре "Крестики-нолики" с применением обучения с подкреплением.

Предоставляет различные режимы работы:
- train: Обучение агента
- evaluate: Оценка эффективности агента
- play: Режим игры человека против агента
- analyze: Анализ стратегии обученного агента
"""

# Стандартные библиотеки
import argparse
import os
import time
from typing import Tuple, Optional, List, Any, Union, Callable

# Библиотеки для работы с данными
import numpy as np
import matplotlib.pyplot as plt

# Локальные модули
from environment import TicTacToeEnv
from agents import QLearningAgent, RandomAgent
from training import train_agent, evaluate_agent, plot_training_stats
from utils import (
    plot_q_values_heatmap,
    visualize_game,
    plot_q_distribution,
    animate_game,
    compare_agents,
    analyze_opponent_types,
    plot_first_move_heatmap,
    learning_curve_analysis
)


def create_env(opponent_type: str = 'random') -> TicTacToeEnv:
    """
    Description:
    ---------------
        Создание среды игры с указанным типом оппонента.
    
    Args:
    ---------------
        opponent_type: Тип оппонента ('random', 'rule_based', 'minimax')
    
    Returns:
    ---------------
        Инициализированная среда игры TicTacToeEnv
    
    Examples:
    ---------------
        >>> env = create_env('rule_based')
        >>> state = env.reset()
    """
    return TicTacToeEnv(opponent_type=opponent_type)


def main() -> None:
    """
    Description:
    ---------------
        Основная функция программы, обрабатывающая аргументы командной
        строки и запускающая соответствующий режим работы.
    
    Returns:
    ---------------
        None
    
    Examples:
    ---------------
        >>> main()  # Запуск с параметрами по умолчанию
    """
    # Настройка парсера аргументов командной строки
    parser = argparse.ArgumentParser(
        description='RL для игры в крестики-нолики'
    )
    
    # Основные параметры
    parser.add_argument(
        '--mode',
        type=str,
        default='train',
        choices=['train', 'evaluate', 'play', 'analyze'],
        help='Режим работы: train, evaluate, play, analyze'
    )
    parser.add_argument(
        '--opponent',
        type=str,
        default='random',
        choices=['random', 'rule_based', 'minimax'],
        help='Тип оппонента'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default='models/q_agent_final.pkl',
        help='Путь к сохраненной модели для загрузки'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=10000,
        help='Количество эпизодов для обучения или оценки'
    )
    parser.add_argument(
        '--render',
        action='store_true',
        help='Включить визуализацию'
    )
    
    # Параметры обучения
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.1,
        help='Скорость обучения'
    )
    parser.add_argument(
        '--discount-factor',
        type=float,
        default=0.9,
        help='Коэффициент дисконтирования'
    )
    parser.add_argument(
        '--exploration-rate',
        type=float,
        default=1.0,
        help='Начальная вероятность исследования'
    )
    parser.add_argument(
        '--exploration-decay',
        type=float,
        default=0.995,
        help='Коэффициент снижения вероятности исследования'
    )
    parser.add_argument(
        '--exploration-min',
        type=float,
        default=0.01,
        help='Минимальная вероятность исследования'
    )
    
    # Парсинг аргументов
    args = parser.parse_args()
    
    # Создание директорий для сохранения моделей и результатов
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Создание среды с указанным типом оппонента
    env = create_env(opponent_type=args.opponent)
    
    # Выбор режима работы
    if args.mode == 'train':
        run_training_mode(args, env)
    elif args.mode == 'evaluate':
        run_evaluation_mode(args, env)
    elif args.mode == 'play':
        run_play_mode(args)
    elif args.mode == 'analyze':
        run_analysis_mode(args, env)
    
    # Закрытие среды
    env.close()


def run_training_mode(args: argparse.Namespace, env: TicTacToeEnv) -> None:
    """
    Description:
    ---------------
        Запуск режима обучения агента.
    
    Args:
    ---------------
        args: Аргументы командной строки
        env: Среда игры
    
    Returns:
    ---------------
        None
    """
    print(f"Начало обучения против оппонента типа '{args.opponent}'")
    
    # Создание агента с указанными параметрами
    agent = QLearningAgent(
        env.action_space,
        learning_rate=args.learning_rate,
        discount_factor=args.discount_factor,
        exploration_rate=args.exploration_rate,
        exploration_decay=args.exploration_decay,
        exploration_min=args.exploration_min
    )
    
    # Обучение агента
    training_stats = train_agent(
        env=env,
        agent=agent,
        n_episodes=args.episodes,
        evaluate_every=args.episodes // 100,  # Оценка каждые 1% эпизодов
        n_eval_episodes=100,
        save_dir='models',
        save_every=args.episodes // 10,  # Сохранение каждые 10% эпизодов
        render_training=args.render
    )
    
    # Визуализация результатов обучения
    plot_training_stats(
        training_stats,
        save_path='results/training_stats.png'
    )
    
    # Анализ кривой обучения
    analysis = learning_curve_analysis(training_stats)
    print("\nАнализ обучения:")
    print(f"Эпизод сходимости: {analysis['convergence_episode']}")
    print(
        f"Максимальный процент побед: {analysis['max_win_rate']:.2f}% "
        f"(эпизод {analysis['max_win_rate_episode']})"
    )
    print(f"Стабильность: {analysis['stability']:.4f}")
    print(
        f"Финальные показатели: "
        f"Победы: {analysis['final_win_rate']:.2f}%, "
        f"Ничьи: {analysis['final_draw_rate']:.2f}%, "
        f"Поражения: {analysis['final_loss_rate']:.2f}%"
    )
    
    # Визуализация распределения Q-значений
    plot_q_distribution(agent, save_path='results/q_distribution.png')
    
    # Визуализация предпочтений для первого хода
    plot_first_move_heatmap(
        agent,
        title='Предпочтения для первого хода',
        save_path='results/first_move_heatmap.png'
    )


def run_evaluation_mode(args: argparse.Namespace, env: TicTacToeEnv) -> None:
    """
    Description:
    ---------------
        Запуск режима оценки обученного агента.
    
    Args:
    ---------------
        args: Аргументы командной строки
        env: Среда игры
    
    Returns:
    ---------------
        None
    """
    print(f"Начало оценки против оппонента типа '{args.opponent}'")
    
    # Создание агента
    agent = QLearningAgent(env.action_space)
    
    # Загрузка модели
    if not agent.load(args.model_path):
        print(f"Ошибка: Не удалось загрузить модель из {args.model_path}")
        return
    
    # Оценка производительности
    eval_stats = evaluate_agent(
        env,
        agent,
        n_episodes=args.episodes,
        render=args.render
    )
    
    print("\nРезультаты оценки:")
    print(f"Количество эпизодов: {args.episodes}")
    print(f"Процент побед: {eval_stats['win_rate'] * 100:.2f}%")
    print(f"Процент ничьих: {eval_stats['draw_rate'] * 100:.2f}%")
    print(f"Процент поражений: {eval_stats['loss_rate'] * 100:.2f}%")
    print(f"Средняя награда: {eval_stats['avg_reward']:.4f}")
    
    # Сравнение с случайным агентом
    random_agent = RandomAgent(env.action_space)
    agents = [agent, random_agent]
    agent_names = ['Q-Learning', 'Random']
    
    comparison = compare_agents(env, agents, agent_names, n_episodes=100)
    print("\nСравнение с случайным агентом:")
    print(comparison)
    
    # Анализ против различных типов оппонентов
    opponent_types = ['random', 'rule_based', 'minimax']
    
    opponent_analysis = analyze_opponent_types(
        env_creator=create_env,
        agent=agent,
        opponent_types=opponent_types,
        n_episodes=100
    )
    
    print("\nАнализ против различных типов оппонентов:")
    print(opponent_analysis)


def run_play_mode(args: argparse.Namespace) -> None:
    """
    Description:
    ---------------
        Запуск режима игры человека против обученного агента.
    
    Args:
    ---------------
        args: Аргументы командной строки
    
    Returns:
    ---------------
        None
    """
    print("Режим игры человека против обученного агента")
    
    # Создание агента
    agent = QLearningAgent(9)  # Предполагаем размер action_space = 9
    
    # Загрузка модели
    if not agent.load(args.model_path):
        print(f"Ошибка: Не удалось загрузить модель из {args.model_path}")
        return
    
    # Инициализация игры
    board = np.zeros((3, 3), dtype=np.int8)
    game_over = False
    
    # Определяем маркеры для игроков
    agent_mark = 1    # X для агента
    human_mark = -1   # O для человека
    
    # Кто ходит первым (можно выбрать или определить случайно)
    current_player = agent_mark  # Агент ходит первым
    
    # Отображение начального состояния
    render_board(board)
    
    # Игровой цикл
    while not game_over:
        if current_player == agent_mark:
            # Ход агента
            valid_actions = [
                i for i, val in enumerate(board.flatten()) if val == 0
            ]
            
            if not valid_actions:
                print("Ничья!")
                break
            
            action = agent.choose_action(
                board, valid_actions, training=False
            )
            row, col = action // 3, action % 3
            
            print(f"Агент выбирает ход: {row}, {col}")
            board[row, col] = agent_mark
            
        else:
            # Ход человека
            valid_actions = [
                i for i, val in enumerate(board.flatten()) if val == 0
            ]
            
            if not valid_actions:
                print("Ничья!")
                break
            
            # Получение и валидация хода человека
            action = get_human_action(board, valid_actions)
            row, col = action // 3, action % 3
            board[row, col] = human_mark
        
        # Отображение доски
        render_board(board)
        
        # Проверка завершения игры
        game_over, winner = check_game_over(board, current_player)
        
        if game_over:
            if winner == agent_mark:
                print("Агент победил!")
            elif winner == human_mark:
                print("Вы победили!")
            else:
                print("Ничья!")
            break
        
        # Смена игрока
        current_player = (
            human_mark if current_player == agent_mark else agent_mark
        )


def run_analysis_mode(args: argparse.Namespace, env: TicTacToeEnv) -> None:
    """
    Description:
    ---------------
        Запуск режима анализа обученного агента.
    
    Args:
    ---------------
        args: Аргументы командной строки
        env: Среда игры
    
    Returns:
    ---------------
        None
    """
    print("Режим анализа обученного агента")
    
    # Создание агента
    agent = QLearningAgent(env.action_space)
    
    # Загрузка модели
    if not agent.load(args.model_path):
        print(f"Ошибка: Не удалось загрузить модель из {args.model_path}")
        return
    
    # Анализ предпочтений для первого хода
    print("Анализ предпочтений для первого хода:")
    plot_first_move_heatmap(agent)
    
    # Анализ Q-значений для различных состояний
    print("Анализ некоторых ключевых состояний:")
    
    # 1. Пустое поле
    empty_board = np.zeros((3, 3), dtype=np.int8)
    plot_q_values_heatmap(agent, empty_board, "Q-значения для пустого поля")
    
    # 2. Первый ход в центр
    center_board = np.zeros((3, 3), dtype=np.int8)
    center_board[1, 1] = 1  # Агент ходит в центр
    plot_q_values_heatmap(
        agent,
        center_board,
        "Q-значения после хода агента в центр"
    )
    
    # 3. Первый ход в угол
    corner_board = np.zeros((3, 3), dtype=np.int8)
    corner_board[0, 0] = 1  # Агент ходит в верхний левый угол
    plot_q_values_heatmap(
        agent,
        corner_board,
        "Q-значения после хода агента в угол"
    )
    
    # 4. Анимация игры
    print("Анимация игры:")
    animate_game(env, agent, max_steps=10)
    
    # 5. Анализ распределения Q-значений
    print("Распределение Q-значений:")
    plot_q_distribution(agent)
    
    # 6. Размер Q-таблицы
    print(f"Размер Q-таблицы: {len(agent.q_table)} состояний")
    
    # 7. Сравнение с другими агентами
    print("Сравнение с случайным агентом:")
    random_agent = RandomAgent(env.action_space)
    agents = [agent, random_agent]
    agent_names = ['Q-Learning', 'Random']
    
    comparison = compare_agents(env, agents, agent_names, n_episodes=100)
    print(comparison)
    
    # 8. Анализ против различных типов оппонентов
    print("Анализ против различных типов оппонентов:")
    opponent_types = ['random', 'rule_based', 'minimax']
    
    opponent_analysis = analyze_opponent_types(
        env_creator=create_env,
        agent=agent,
        opponent_types=opponent_types,
        n_episodes=100
    )
    
    print(opponent_analysis)


def render_board(board: np.ndarray) -> None:
    """
    Description:
    ---------------
        Отображение игрового поля в консоли.
    
    Args:
    ---------------
        board: Текущее состояние игрового поля (матрица 3x3)
    
    Returns:
    ---------------
        None
    """
    symbols = {0: " ", 1: "X", -1: "O"}
    board_str = "\n"
    
    for row in range(3):
        for col in range(3):
            board_str += f" {symbols[board[row, col]]} "
            if col < 2:
                board_str += "|"
        
        if row < 2:
            board_str += "\n---+---+---\n"
        else:
            board_str += "\n"
    
    print(board_str)


def get_human_action(
    board: np.ndarray,
    valid_actions: List[int]
) -> int:
    """
    Description:
    ---------------
        Получение и валидация хода человека.
    
    Args:
    ---------------
        board: Текущее состояние игрового поля
        valid_actions: Список допустимых действий
    
    Returns:
    ---------------
        Валидное действие (индекс от 0 до 8)
    """
    while True:
        try:
            row = int(input("Введите номер строки (0-2): "))
            col = int(input("Введите номер столбца (0-2): "))
            
            if row < 0 or row > 2 or col < 0 or col > 2:
                print("Недопустимые координаты. Попробуйте снова.")
                continue
            
            action = row * 3 + col
            
            if action not in valid_actions:
                print("Клетка уже занята. Попробуйте снова.")
                continue
            
            return action
        except ValueError:
            print("Необходимо ввести число. Попробуйте снова.")


def check_game_over(
    board: np.ndarray,
    mark: int
) -> Tuple[bool, Optional[int]]:
    """
    Description:
    ---------------
        Проверка завершения игры.
    
    Args:
    ---------------
        board: Текущее состояние игрового поля
        mark: Маркер текущего игрока (1 или -1)
    
    Returns:
    ---------------
        Кортеж из флага завершения игры и победителя (None для ничьей)
    """
    # Проверка горизонтальных линий
    for row in range(3):
        if np.sum(board[row, :]) == mark * 3:
            return True, mark
    
    # Проверка вертикальных линий
    for col in range(3):
        if np.sum(board[:, col]) == mark * 3:
            return True, mark
    
    # Проверка диагоналей
    if np.trace(board) == mark * 3 or np.trace(np.fliplr(board)) == mark * 3:
        return True, mark
    
    # Проверка на ничью
    if np.all(board != 0):
        return True, None
    
    return False, None


if __name__ == "__main__":
    main()