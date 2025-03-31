#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Модуль, реализующий среду OpenAI Gym для игры в крестики-нолики.
"""

# Стандартные библиотеки
from typing import Dict, Optional, Tuple, Union, Any

# Сторонние библиотеки
import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt

# Локальные импорты
from .opponent import RandomOpponent, RuleBasedOpponent, MinimaxOpponent


class TicTacToeEnv(gym.Env):
    """
    Description:
    ---------------
        Среда для игры в крестики-нолики на основе OpenAI Gym.
        Поддерживает различные типы оппонентов и визуализацию.
    
    Attributes:
    ---------------
        observation_space: Пространство наблюдений (3x3 поле)
        action_space: Пространство действий (9 возможных клеток)
        board: Текущее состояние игрового поля
        opponent: Выбранный тип оппонента
        agent_mark: Маркер хода агента (1 - X)
        opponent_mark: Маркер хода оппонента (-1 - O)
        current_player: Текущий игрок (1 - агент, -1 - оппонент)
        done: Флаг завершения игры
        winner: Победитель игры (1 - агент, -1 - оппонент, None - ничья)
        info: Дополнительная информация о состоянии игры
    """
    
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self, opponent_type: str = 'random') -> None:
        """
        Description:
        ---------------
            Инициализация среды.
        
        Args:
        ---------------
            opponent_type: Тип оппонента ('random', 'rule_based', 'minimax')
        
        Raises:
        ---------------
            ValueError: Если указан неизвестный тип оппонента
        
        Examples:
        ---------------
            >>> env = TicTacToeEnv(opponent_type='random')
            >>> observation = env.reset()
        """
        # Определение пространств наблюдения (3x3 поле) и действий (9 возможных клеток)
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(3, 3), dtype=np.int8
        )
        self.action_space = spaces.Discrete(9)
        
        # Создание поля 3x3, заполненного нулями (пустыми клетками)
        self.board = np.zeros((3, 3), dtype=np.int8)
        
        # Создание соответствующего оппонента
        if opponent_type == 'random':
            self.opponent = RandomOpponent()
        elif opponent_type == 'rule_based':
            self.opponent = RuleBasedOpponent()
        elif opponent_type == 'minimax':
            self.opponent = MinimaxOpponent()
        else:
            raise ValueError(f"Неизвестный тип оппонента: {opponent_type}")
        
        # Маркеры ходов: 1 для агента (X), -1 для оппонента (O)
        self.agent_mark = 1
        self.opponent_mark = -1
        
        # Инициализация переменных для отслеживания состояния игры
        self.current_player = None  # Кто сейчас ходит (1: агент, -1: оппонент)
        self.done = False
        self.winner = None
        self.info = {}
        
        # Инициализация для отрисовки
        self.fig = None
        self.ax = None
    
    def reset(self) -> np.ndarray:
        """
        Description:
        ---------------
            Сброс среды в начальное состояние.
        
        Returns:
        ---------------
            observation: Начальное состояние игрового поля (матрица 3x3)
        
        Examples:
        ---------------
            >>> env = TicTacToeEnv()
            >>> observation = env.reset()
            >>> print(observation)
            [[0 0 0]
             [0 0 0]
             [0 0 0]]
        """
        # Сброс поля
        self.board = np.zeros((3, 3), dtype=np.int8)
        
        # Сброс переменных состояния
        self.current_player = self.agent_mark  # Агент ходит первым
        self.done = False
        self.winner = None
        self.info = {}
        
        # Закрытие предыдущего графика, если он был
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
        
        return self.board.copy()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Description:
        ---------------
            Выполнение хода агента и ответного хода оппонента.
        
        Args:
        ---------------
            action: Выбранная агентом клетка (0-8)
            
        Returns:
        ---------------
            observation: Новое состояние поля (матрица 3x3)
            reward: Полученная награда
            done: Флаг завершения игры
            info: Дополнительная информация
        
        Examples:
        ---------------
            >>> env = TicTacToeEnv()
            >>> env.reset()
            >>> observation, reward, done, info = env.step(4)  # Ход в центр поля
        """
        # Проверка, что игра не завершена
        if self.done:
            return self.board.copy(), 0.0, self.done, self.info
        
        # Преобразование линейного индекса (0-8) в координаты (строка, столбец)
        row, col = action // 3, action % 3
        
        # Проверка валидности хода
        if self.board[row, col] != 0:
            # Недопустимый ход: пытаемся выбрать уже занятую клетку
            return self.board.copy(), -1.0, True, {"message": "Недопустимый ход"}
        
        # Выполнение хода агента
        self.board[row, col] = self.agent_mark
        
        # Проверка, не завершилась ли игра после хода агента
        if self._check_game_over():
            # Определение награды на основе результата
            reward = self._get_reward()
            return self.board.copy(), reward, self.done, self.info
        
        # Ход оппонента
        opponent_action = self.opponent.choose_action(
            self.board, self.opponent_mark, self.agent_mark
        )
        
        # Если оппонент вернул None, значит нет доступных ходов (ничья)
        if opponent_action is None:
            self.done = True
            self.info = {"message": "Ничья", "winner": None}
            return self.board.copy(), 0.0, self.done, self.info
        
        # Преобразование выбранного хода оппонента в координаты
        opp_row, opp_col = opponent_action // 3, opponent_action % 3
        
        # Выполнение хода оппонента
        self.board[opp_row, opp_col] = self.opponent_mark
        
        # Проверка, не завершилась ли игра после хода оппонента
        self._check_game_over()
        
        # Определение награды на основе результата
        reward = self._get_reward()
        
        # Небольшой штраф за каждый ход, чтобы стимулировать быстрые победы
        if not self.done:
            reward -= 0.01
        
        return self.board.copy(), reward, self.done, self.info
    
    def _check_game_over(self) -> bool:
        """
        Description:
        ---------------
            Проверка условий окончания игры.
        
        Returns:
        ---------------
            Возвращает True, если игра завершена, иначе False
        
        Examples:
        ---------------
            >>> env = TicTacToeEnv()
            >>> env.board = np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0]])
            >>> env._check_game_over()
            True
        """
        # Проверка горизонтальных линий
        for row in range(3):
            if abs(np.sum(self.board[row, :])) == 3:
                self.done = True
                self.winner = (
                    self.agent_mark if np.sum(self.board[row, :]) == 3 
                    else self.opponent_mark
                )
                self.info = {
                    "message": "Победа" if self.winner == self.agent_mark else "Поражение", 
                    "winner": self.winner
                }
                return True
        
        # Проверка вертикальных линий
        for col in range(3):
            if abs(np.sum(self.board[:, col])) == 3:
                self.done = True
                self.winner = (
                    self.agent_mark if np.sum(self.board[:, col]) == 3 
                    else self.opponent_mark
                )
                self.info = {
                    "message": "Победа" if self.winner == self.agent_mark else "Поражение", 
                    "winner": self.winner
                }
                return True
        
        # Проверка диагоналей
        diag_sum = np.trace(self.board)
        anti_diag_sum = np.trace(np.fliplr(self.board))
        
        if abs(diag_sum) == 3:
            self.done = True
            self.winner = (
                self.agent_mark if diag_sum == 3 else self.opponent_mark
            )
            self.info = {
                "message": "Победа" if self.winner == self.agent_mark else "Поражение", 
                "winner": self.winner
            }
            return True
            
        if abs(anti_diag_sum) == 3:
            self.done = True
            self.winner = (
                self.agent_mark if anti_diag_sum == 3 else self.opponent_mark
            )
            self.info = {
                "message": "Победа" if self.winner == self.agent_mark else "Поражение", 
                "winner": self.winner
            }
            return True
        
        # Проверка на ничью (все клетки заполнены)
        if np.all(self.board != 0):
            self.done = True
            self.winner = None
            self.info = {"message": "Ничья", "winner": None}
            return True
        
        return False
    
    def _get_reward(self) -> float:
        """
        Description:
        ---------------
            Вычисление награды на основе состояния игры.
        
        Returns:
        ---------------
            Значение награды:
            1.0 - агент победил
            -1.0 - агент проиграл
            0.0 - ничья или игра не завершена
        
        Examples:
        ---------------
            >>> env = TicTacToeEnv()
            >>> env.done = True
            >>> env.winner = env.agent_mark
            >>> env._get_reward()
            1.0
        """
        if not self.done:
            return 0.0
        
        if self.winner == self.agent_mark:
            return 1.0  # Агент победил
        elif self.winner == self.opponent_mark:
            return -1.0  # Агент проиграл
        else:
            return 0.0  # Ничья
    
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """
        Description:
        ---------------
            Визуализация игрового поля.
        
        Args:
        ---------------
            mode: Режим отображения ('human', 'rgb_array')
            
        Returns:
        ---------------
            Если mode='rgb_array', возвращает RGB представление поля,
            иначе None
        
        Examples:
        ---------------
            >>> env = TicTacToeEnv()
            >>> env.reset()
            >>> env.render()  # Выводит пустое поле в консоль
        """
        if mode == 'human':
            # Текстовое представление для вывода в консоль
            board_str = "\n"
            for row in range(3):
                for col in range(3):
                    if self.board[row, col] == 0:
                        board_str += "   "
                    elif self.board[row, col] == 1:
                        board_str += " X "
                    else:
                        board_str += " O "
                    
                    if col < 2:
                        board_str += "|"
                
                if row < 2:
                    board_str += "\n---+---+---\n"
                else:
                    board_str += "\n"
            
            print(board_str)
            return None
            
        elif mode == 'rgb_array':
            # Графическое представление с использованием matplotlib
            if self.fig is None or self.ax is None:
                self.fig, self.ax = plt.subplots(figsize=(6, 6))
                self.ax.set_xlim([0, 3])
                self.ax.set_ylim([0, 3])
                self.ax.set_xticks([1, 2, 3])
                self.ax.set_yticks([1, 2, 3])
                self.ax.grid(True)
                self.ax.set_title("Крестики-нолики")
                
            # Очистка предыдущего состояния
            self.ax.clear()
            self.ax.set_xlim([0, 3])
            self.ax.set_ylim([0, 3])
            self.ax.grid(True)
            
            # Отрисовка сетки
            for i in range(4):
                self.ax.axhline(i, color='black', linewidth=1)
                self.ax.axvline(i, color='black', linewidth=1)
            
            # Отрисовка X и O
            for row in range(3):
                for col in range(3):
                    if self.board[row, col] == 1:  # X
                        self.ax.plot(
                            [col + 0.2, col + 0.8], 
                            [2 - row + 0.2, 2 - row + 0.8], 
                            'r-', 
                            linewidth=2
                        )
                        self.ax.plot(
                            [col + 0.8, col + 0.2], 
                            [2 - row + 0.2, 2 - row + 0.8], 
                            'r-', 
                            linewidth=2
                        )
                    elif self.board[row, col] == -1:  # O
                        circle = plt.Circle(
                            (col + 0.5, 2 - row + 0.5), 
                            0.3, 
                            fill=False, 
                            color='blue', 
                            linewidth=2
                        )
                        self.ax.add_artist(circle)
            
            plt.tight_layout()
            plt.draw()
            plt.pause(0.1)
            
            # Возвращаем RGB представление
            self.fig.canvas.draw()
            image = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            image = image.reshape(
                self.fig.canvas.get_width_height()[::-1] + (3,)
            )
            return image
        
        return None
    
    def close(self) -> None:
        """
        Description:
        ---------------
            Закрытие среды и освобождение ресурсов.
        
        Examples:
        ---------------
            >>> env = TicTacToeEnv()
            >>> env.reset()
            >>> env.close()  # Закрытие всех ресурсов среды
        """
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None