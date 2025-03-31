#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Стандартные библиотеки
import random

# Библиотеки для научных вычислений
import numpy as np

# Библиотеки для типизации
from typing import List, Optional, Any


class RandomAgent:
    """
    Description:
    ---------------
        Агент, выбирающий случайные действия из списка допустимых.
        Используется как базовый алгоритм или для сравнения с обучаемыми агентами.
    
    Args:
    ---------------
        action_space: Пространство действий
    
    Returns:
    ---------------
        Объект агента, выбирающего случайные действия
    
    Examples:
    ---------------
        >>> agent = RandomAgent(action_space=range(9))
        >>> action = agent.choose_action(state)
    """
    
    def __init__(self, action_space: List[int]) -> None:
        """
        Description:
        ---------------
            Инициализация агента с случайной стратегией.
        
        Args:
        ---------------
            action_space: Пространство действий
        
        Returns:
        ---------------
            None
        
        Examples:
        ---------------
            >>> agent = RandomAgent(action_space=range(9))
        """
        self.action_space = action_space
    
    def choose_action(
        self, 
        state: np.ndarray, 
        valid_actions: Optional[List[int]] = None, 
        training: bool = True
    ) -> int:
        """
        Description:
        ---------------
            Выбор случайного допустимого действия из списка валидных действий.
            Параметр training игнорируется, так как этот агент не обучается.
        
        Args:
        ---------------
            state: Текущее состояние среды (игровое поле)
            valid_actions: Список допустимых действий (по умолчанию - пустые клетки)
            training: Флаг режима обучения (игнорируется для данного агента)
            
        Returns:
        ---------------
            Индекс выбранного действия
        
        Examples:
        ---------------
            >>> action = agent.choose_action(state, [0, 1, 2])
            >>> # Вернет случайный элемент из [0, 1, 2]
        """
        # Если valid_actions не передан, находим пустые клетки на поле
        if valid_actions is None:
            # Преобразуем двумерный массив в одномерный для поиска пустых клеток
            flat_state = state.flatten()
            # Определяем индексы пустых клеток (значение 0)
            valid_actions = [i for i, val in enumerate(flat_state) if val == 0]
        
        # Выбор случайного действия из доступных
        # Используем random.choice для равномерного выбора из списка
        return random.choice(valid_actions)
    
    def learn(
        self, 
        state: np.ndarray, 
        action: int, 
        reward: float, 
        next_state: np.ndarray, 
        done: bool
    ) -> None:
        """
        Description:
        ---------------
            Пустая функция для совместимости с интерфейсом обучаемых агентов.
            Этот агент не обучается, поэтому метод ничего не делает.
        
        Args:
        ---------------
            state: Текущее состояние среды перед выполнением действия
            action: Выполненное действие
            reward: Полученная награда
            next_state: Следующее состояние после выполнения действия
            done: Флаг завершения эпизода
        
        Returns:
        ---------------
            None
        
        Examples:
        ---------------
            >>> agent.learn(state, 4, 1.0, next_state, False)
            >>> # Ничего не происходит, так как агент не обучается
        """
        # Этот агент не обучается, поэтому функция пустая
        pass