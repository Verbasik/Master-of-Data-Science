#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Модуль, реализующий различные типы оппонентов для игры в крестики-нолики.
Включает базовый абстрактный класс и три конкретные реализации оппонентов
с различными стратегиями.
"""

# Стандартные библиотеки
import random
from abc import ABC, abstractmethod
from typing import List, Optional

# Сторонние библиотеки
import numpy as np


class Opponent(ABC):
    """
    Description:
    ---------------
        Базовый абстрактный класс для оппонентов в игре крестики-нолики.
        Определяет общий интерфейс и вспомогательные методы.
        
    Attributes:
    ---------------
        Не содержит атрибутов, только методы.
    """
    
    @abstractmethod
    def choose_action(
        self, 
        board: np.ndarray, 
        opponent_mark: int, 
        agent_mark: int
    ) -> Optional[int]:
        """
        Description:
        ---------------
            Выбор хода для оппонента.
        
        Args:
        ---------------
            board: Текущее состояние игрового поля
            opponent_mark: Маркер оппонента (обычно -1)
            agent_mark: Маркер агента (обычно 1)
            
        Returns:
        ---------------
            Выбранный ход (0-8) или None, если нет доступных ходов
            
        Examples:
        ---------------
            >>> op = ConcreteOpponent()
            >>> board = np.zeros((3, 3), dtype=np.int8)
            >>> op.choose_action(board, -1, 1)
            4  # Например, выбран центр поля
        """
        pass
    
    def _get_valid_moves(self, board: np.ndarray) -> List[int]:
        """
        Description:
        ---------------
            Получение списка допустимых ходов на текущем поле.
        
        Args:
        ---------------
            board: Текущее состояние игрового поля
            
        Returns:
        ---------------
            Список допустимых ходов (линейные индексы 0-8)
            
        Examples:
        ---------------
            >>> op = ConcreteOpponent()
            >>> board = np.array([[1, 0, -1], [0, 0, 0], [0, 0, 0]])
            >>> op._get_valid_moves(board)
            [1, 3, 4, 5, 6, 7, 8]  # Все свободные клетки
        """
        flat_board = board.flatten()
        return [i for i, cell in enumerate(flat_board) if cell == 0]


class RandomOpponent(Opponent):
    """
    Description:
    ---------------
        Оппонент, выбирающий случайные ходы из доступных.
        Самая простая стратегия, применяемая для базового обучения агента.
        
    Attributes:
    ---------------
        Наследует атрибуты базового класса Opponent.
    """
    
    def choose_action(
        self, 
        board: np.ndarray, 
        opponent_mark: int, 
        agent_mark: int
    ) -> Optional[int]:
        """
        Description:
        ---------------
            Выбор случайного хода из доступных.
        
        Args:
        ---------------
            board: Текущее состояние игрового поля
            opponent_mark: Маркер оппонента (обычно -1)
            agent_mark: Маркер агента (обычно 1)
            
        Returns:
        ---------------
            Выбранный ход (0-8) или None, если нет доступных ходов
            
        Examples:
        ---------------
            >>> op = RandomOpponent()
            >>> board = np.zeros((3, 3), dtype=np.int8)
            >>> action = op.choose_action(board, -1, 1)
            >>> 0 <= action <= 8
            True
        """
        valid_moves = self._get_valid_moves(board)
        
        if not valid_moves:
            return None
        
        return random.choice(valid_moves)


class RuleBasedOpponent(Opponent):
    """
    Description:
    ---------------
        Оппонент, использующий простые правила для выбора хода.
        Применяет стратегию, основанную на приоритетах ходов.
        
    Attributes:
    ---------------
        Наследует атрибуты базового класса Opponent.
    """
    
    def choose_action(
        self, 
        board: np.ndarray, 
        opponent_mark: int, 
        agent_mark: int
    ) -> Optional[int]:
        """
        Description:
        ---------------
            Выбор хода по правилам:
            1. Завершить победную комбинацию
            2. Блокировать победную комбинацию агента
            3. Занять центр, если доступен
            4. Занять угол, если доступен
            5. Занять сторону
        
        Args:
        ---------------
            board: Текущее состояние игрового поля
            opponent_mark: Маркер оппонента (обычно -1)
            agent_mark: Маркер агента (обычно 1)
            
        Returns:
        ---------------
            Выбранный ход (0-8) или None, если нет доступных ходов
            
        Examples:
        ---------------
            >>> op = RuleBasedOpponent()
            >>> board = np.array([[-1, -1, 0], [0, 1, 0], [1, 0, 0]])
            >>> op.choose_action(board, -1, 1)
            2  # Завершает победную комбинацию по горизонтали
        """
        valid_moves = self._get_valid_moves(board)
        
        if not valid_moves:
            return None
        
        # 1. Проверка возможности победы
        for move in valid_moves:
            row, col = move // 3, move % 3
            board_copy = board.copy()
            board_copy[row, col] = opponent_mark
            
            if self._check_win(board_copy, opponent_mark):
                return move
        
        # 2. Блокирование потенциальной победы агента
        for move in valid_moves:
            row, col = move // 3, move % 3
            board_copy = board.copy()
            board_copy[row, col] = agent_mark
            
            if self._check_win(board_copy, agent_mark):
                return move
        
        # 3. Занять центр
        if 4 in valid_moves:
            return 4
        
        # 4. Занять угол
        corners = [0, 2, 6, 8]
        available_corners = [move for move in corners if move in valid_moves]
        if available_corners:
            return random.choice(available_corners)
        
        # 5. Занять сторону
        sides = [1, 3, 5, 7]
        available_sides = [move for move in sides if move in valid_moves]
        if available_sides:
            return random.choice(available_sides)
        
        # Если дошли до этого места, просто выбираем случайный ход
        return random.choice(valid_moves)
    
    def _check_win(self, board: np.ndarray, mark: int) -> bool:
        """
        Description:
        ---------------
            Проверка, есть ли победная комбинация для указанного маркера.
        
        Args:
        ---------------
            board: Состояние игрового поля
            mark: Маркер для проверки
            
        Returns:
        ---------------
            True, если есть победная комбинация, иначе False
            
        Examples:
        ---------------
            >>> op = RuleBasedOpponent()
            >>> board = np.array([[1, 1, 1], [0, -1, 0], [0, 0, -1]])
            >>> op._check_win(board, 1)
            True
        """
        # Проверка горизонтальных линий
        for row in range(3):
            if np.sum(board[row, :]) == mark * 3:
                return True
        
        # Проверка вертикальных линий
        for col in range(3):
            if np.sum(board[:, col]) == mark * 3:
                return True
        
        # Проверка диагоналей
        if np.trace(board) == mark * 3 or np.trace(np.fliplr(board)) == mark * 3:
            return True
        
        return False


class MinimaxOpponent(Opponent):
    """
    Description:
    ---------------
        Оппонент, использующий алгоритм минимакс для оптимальной игры.
        Эта реализация всегда выбирает наилучший возможный ход, что делает
        её самым сложным оппонентом.
        
    Attributes:
    ---------------
        Наследует атрибуты базового класса Opponent.
    """
    
    def choose_action(
        self, 
        board: np.ndarray, 
        opponent_mark: int, 
        agent_mark: int
    ) -> Optional[int]:
        """
        Description:
        ---------------
            Выбор наилучшего хода с использованием алгоритма минимакс.
        
        Args:
        ---------------
            board: Текущее состояние игрового поля
            opponent_mark: Маркер оппонента (обычно -1)
            agent_mark: Маркер агента (обычно 1)
            
        Returns:
        ---------------
            Выбранный ход (0-8) или None, если нет доступных ходов
            
        Examples:
        ---------------
            >>> op = MinimaxOpponent()
            >>> board = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
            >>> op.choose_action(board, -1, 1)
            4  # Оптимальный ход - центр
        """
        valid_moves = self._get_valid_moves(board)
        
        if not valid_moves:
            return None
        
        best_score = float('-inf')
        best_move = valid_moves[0]  # По умолчанию первый доступный ход
        
        for move in valid_moves:
            row, col = move // 3, move % 3
            board_copy = board.copy()
            board_copy[row, col] = opponent_mark
            
            score = self._minimax(
                board_copy, 
                0, 
                False, 
                opponent_mark, 
                agent_mark
            )
            
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move
    
    def _minimax(
        self, 
        board: np.ndarray, 
        depth: int, 
        is_maximizing: bool, 
        opponent_mark: int, 
        agent_mark: int
    ) -> int:
        """
        Description:
        ---------------
            Рекурсивная функция минимакс для оценки ходов.
        
        Args:
        ---------------
            board: Текущее состояние игрового поля
            depth: Глубина рекурсии
            is_maximizing: True, если максимизирующий игрок, иначе False
            opponent_mark: Маркер оппонента (обычно -1)
            agent_mark: Маркер агента (обычно 1)
            
        Returns:
        ---------------
            Оценка текущего состояния
            
        Examples:
        ---------------
            >>> op = MinimaxOpponent()
            >>> board = np.array([[1, 1, 0], [0, -1, 0], [0, 0, 0]])
            >>> op._minimax(board, 0, True, -1, 1)
            9  # Оценка состояния для оппонента
        """
        # Проверка победы оппонента
        if self._check_win(board, opponent_mark):
            return 10 - depth
        
        # Проверка победы агента
        if self._check_win(board, agent_mark):
            return depth - 10
        
        # Проверка ничьей
        if 0 not in board:
            return 0
        
        valid_moves = self._get_valid_moves(board)
        
        if is_maximizing:
            # Ход оппонента (максимизация)
            best_score = float('-inf')
            
            for move in valid_moves:
                row, col = move // 3, move % 3
                board_copy = board.copy()
                board_copy[row, col] = opponent_mark
                
                score = self._minimax(
                    board_copy, 
                    depth + 1, 
                    False, 
                    opponent_mark, 
                    agent_mark
                )
                best_score = max(score, best_score)
            
            return best_score
        else:
            # Ход агента (минимизация)
            best_score = float('inf')
            
            for move in valid_moves:
                row, col = move // 3, move % 3
                board_copy = board.copy()
                board_copy[row, col] = agent_mark
                
                score = self._minimax(
                    board_copy, 
                    depth + 1, 
                    True, 
                    opponent_mark, 
                    agent_mark
                )
                best_score = min(score, best_score)
            
            return best_score
    
    def _check_win(self, board: np.ndarray, mark: int) -> bool:
        """
        Description:
        ---------------
            Проверка, есть ли победная комбинация для указанного маркера.
        
        Args:
        ---------------
            board: Состояние игрового поля
            mark: Маркер для проверки
            
        Returns:
        ---------------
            True, если есть победная комбинация, иначе False
            
        Examples:
        ---------------
            >>> op = MinimaxOpponent()
            >>> board = np.array([[-1, -1, -1], [0, 1, 0], [1, 0, 0]])
            >>> op._check_win(board, -1)
            True
        """
        # Проверка горизонтальных линий
        for row in range(3):
            if np.sum(board[row, :]) == mark * 3:
                return True
        
        # Проверка вертикальных линий
        for col in range(3):
            if np.sum(board[:, col]) == mark * 3:
                return True
        
        # Проверка диагоналей
        if np.trace(board) == mark * 3 or np.trace(np.fliplr(board)) == mark * 3:
            return True
        
        return False