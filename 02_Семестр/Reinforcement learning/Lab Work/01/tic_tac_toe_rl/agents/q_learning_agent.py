#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Стандартные библиотеки
import os
import pickle
import random

# Библиотеки для научных вычислений
import numpy as np

# Библиотеки для типизации
from typing import Dict, List, Optional, Tuple, Union, Any


class QLearningAgent:
    """
    Description:
    ---------------
        Агент на основе алгоритма Q-learning для обучения с подкреплением.
        Использует Q-таблицу для хранения и обновления оценок действий в различных состояниях.
    
    Args:
    ---------------
        action_space: Пространство действий
        learning_rate: Скорость обучения (альфа)
        discount_factor: Коэффициент дисконтирования (гамма)
        exploration_rate: Начальная вероятность исследования (эпсилон)
        exploration_decay: Коэффициент снижения вероятности исследования
        exploration_min: Минимальная вероятность исследования
    
    Returns:
    ---------------
        Объект агента Q-learning
    
    Examples:
    ---------------
        >>> agent = QLearningAgent(action_space=range(9))
        >>> action = agent.choose_action(state)
        >>> agent.learn(state, action, reward, next_state, done)
    """
    
    def __init__(
        self, 
        action_space: List[int], 
        learning_rate: float = 0.1, 
        discount_factor: float = 0.9,
        exploration_rate: float = 1.0, 
        exploration_decay: float = 0.995, 
        exploration_min: float = 0.01
    ) -> None:
        """
        Description:
        ---------------
            Инициализация агента Q-learning.
        
        Args:
        ---------------
            action_space: Пространство действий
            learning_rate: Скорость обучения (альфа)
            discount_factor: Коэффициент дисконтирования (гамма)
            exploration_rate: Начальная вероятность исследования (эпсилон)
            exploration_decay: Коэффициент снижения вероятности исследования
            exploration_min: Минимальная вероятность исследования
        
        Returns:
        ---------------
            None
        
        Examples:
        ---------------
            >>> agent = QLearningAgent(action_space=range(9))
        """
        # Параметры обучения
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.exploration_min = exploration_min
        
        # Инициализация Q-таблицы как словаря
        # Ключ: строковое представление состояния, значение: массив Q-значений для каждого действия
        self.q_table: Dict[str, np.ndarray] = {}
        
        # Счетчики для сбора статистики обучения
        self.episode_count = 0
        self.total_rewards: List[float] = []
        
    def choose_action(
        self, 
        state: np.ndarray, 
        valid_actions: Optional[List[int]] = None, 
        training: bool = True
    ) -> int:
        """
        Description:
        ---------------
            Выбор действия по epsilon-greedy стратегии. В режиме обучения 
            с вероятностью epsilon выбирается случайное действие для исследования,
            иначе выбирается действие с максимальным Q-значением.
        
        Args:
        ---------------
            state: Текущее состояние среды (игровое поле)
            valid_actions: Список допустимых действий (по умолчанию - пустые клетки)
            training: Флаг режима обучения (True - используется exploration)
        
        Returns:
        ---------------
            Индекс выбранного действия (0-8 для игры в крестики-нолики)
        
        Examples:
        ---------------
            >>> action = agent.choose_action(state, [0, 1, 2], True)
            >>> # Вернет индекс действия из допустимых [0, 1, 2]
        """
        # Если valid_actions не передан, считаем допустимыми пустые клетки (0)
        if valid_actions is None:
            flat_state = state.flatten()
            valid_actions = [i for i, val in enumerate(flat_state) if val == 0]
        
        # Преобразование состояния в строковый ключ для Q-таблицы
        state_key = self._get_state_key(state)
        
        # Инициализация Q-значений для нового состояния, если его нет в таблице
        if state_key not in self.q_table:
            # Создаем нулевой вектор для всех 9 действий
            self.q_table[state_key] = np.zeros(9)
            
            # Установка недопустимых действий в -бесконечность,
            # чтобы они никогда не выбирались в качестве лучших
            for action in range(9):
                if action not in valid_actions:
                    self.q_table[state_key][action] = -np.inf
        
        # Применение epsilon-greedy стратегии в режиме обучения
        if training and random.random() < self.exploration_rate:
            # Случайный выбор из допустимых действий для исследования
            return random.choice(valid_actions)
        else:
            # Выбор действия с максимальным Q-значением среди допустимых
            # Сначала создаем копию Q-значений для текущего состояния
            q_values = self.q_table[state_key].copy()
            
            # Маскируем недопустимые действия большим отрицательным числом
            for action in range(9):
                if action not in valid_actions:
                    q_values[action] = -np.inf
            
            # Находим максимальное Q-значение среди допустимых действий
            max_q = np.max(q_values)
            
            # Если все Q-значения одинаковы, выбираем случайное действие
            # (это важно на начальных этапах обучения)
            if np.all(q_values[valid_actions] == q_values[valid_actions][0]):
                return random.choice(valid_actions)
            
            # Иначе выбираем случайное действие среди тех, что имеют максимальное Q-значение
            # (для разрешения конфликтов при одинаковых максимальных значениях)
            actions_with_max_q = [a for a in valid_actions if q_values[a] == max_q]
            return random.choice(actions_with_max_q)
    
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
            Обновление Q-значений на основе полученного опыта по формуле Q-learning:
            Q(s,a) = Q(s,a) + alpha * (r + gamma * max(Q(s',a')) - Q(s,a))
        
        Args:
        ---------------
            state: Текущее состояние перед выполнением действия
            action: Выполненное действие (0-8)
            reward: Полученная награда
            next_state: Следующее состояние после выполнения действия
            done: Флаг завершения эпизода
        
        Returns:
        ---------------
            None
        
        Examples:
        ---------------
            >>> agent.learn(state, 4, 1.0, next_state, False)
            >>> # Обновит Q-значение для действия 4 в состоянии state
        """
        # Получение ключей для текущего и следующего состояний
        state_key = self._get_state_key(state)
        next_state_key = self._get_state_key(next_state)
        
        # Инициализация Q-значений для текущего состояния, если его нет в таблице
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(9)
        
        # Инициализация Q-значений для следующего состояния, если его нет в таблице
        # и эпизод не завершен
        if next_state_key not in self.q_table and not done:
            self.q_table[next_state_key] = np.zeros(9)
            
            # Установка недопустимых действий в -бесконечность для следующего состояния
            flat_next_state = next_state.flatten()
            for a in range(9):
                # Если клетка занята (не 0), делаем действие недопустимым
                if flat_next_state[a] != 0:
                    self.q_table[next_state_key][a] = -np.inf
        
        # Получение максимального Q-значения для следующего состояния
        # (используется в формуле обновления)
        max_next_q = 0
        if not done:
            # Создаем маску для допустимых действий (пустые клетки)
            mask = next_state.flatten() == 0
            valid_actions = [i for i, val in enumerate(mask) if val]
            
            # Находим максимальное Q-значение среди допустимых действий
            if valid_actions:
                max_next_q = max([
                    self.q_table[next_state_key][a] for a in valid_actions
                ])
        
        # Обновление Q-значения по формуле Q-learning
        current_q = self.q_table[state_key][action]
        # Q(s,a) = Q(s,a) + alpha * (r + gamma * max(Q(s',a')) - Q(s,a))
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        self.q_table[state_key][action] = new_q
        
        # Уменьшаем exploration_rate (вероятность исследования), если эпизод завершен
        if done:
            self.exploration_rate = max(
                self.exploration_min, 
                self.exploration_rate * self.exploration_decay
            )
            # Обновляем счетчики для статистики
            self.episode_count += 1
            self.total_rewards.append(reward)
    
    def _get_state_key(self, state: np.ndarray) -> str:
        """
        Description:
        ---------------
            Преобразование состояния (игрового поля) в строковый ключ для Q-таблицы.
            Используется для индексации состояний в словаре Q-таблицы.
        
        Args:
        ---------------
            state: Состояние игрового поля (матрица 3x3 или другой формы)
        
        Returns:
        ---------------
            Строковый ключ, представляющий состояние
        
        Examples:
        ---------------
            >>> key = agent._get_state_key(np.array([[0, 1, 2], [1, 0, 2], [0, 1, 0]]))
            >>> # Вернет '[0, 1, 2, 1, 0, 2, 0, 1, 0]'
        """
        # Преобразуем матрицу состояния в одномерный список и затем в строку
        return str(state.flatten().tolist())
    
    def get_stats(self) -> Dict[str, Union[int, float, List[float]]]:
        """
        Description:
        ---------------
            Получение статистики обучения агента, включая количество эпизодов,
            текущую вероятность исследования, размер Q-таблицы и историю наград.
        
        Args:
        ---------------
            Нет аргументов
        
        Returns:
        ---------------
            Словарь со статистическими данными обучения
        
        Examples:
        ---------------
            >>> stats = agent.get_stats()
            >>> print(f"Количество эпизодов: {stats['episode_count']}")
        """
        return {
            'episode_count': self.episode_count,
            'exploration_rate': self.exploration_rate,
            'q_table_size': len(self.q_table),
            'total_rewards': self.total_rewards
        }
    
    def save(self, filepath: str) -> None:
        """
        Description:
        ---------------
            Сохранение состояния Q-агента в файл с использованием pickle.
            Сохраняются Q-таблица, текущая вероятность исследования и статистика.
        
        Args:
        ---------------
            filepath: Путь к файлу для сохранения
        
        Returns:
        ---------------
            None
        
        Raises:
        ---------------
            IOError: При проблемах с записью в файл
        
        Examples:
        ---------------
            >>> agent.save('q_agent.pkl')
            >>> # Сохранит состояние агента в файл q_agent.pkl
        """
        with open(filepath, 'wb') as f:
            pickle.dump({
                'q_table': self.q_table,
                'exploration_rate': self.exploration_rate,
                'episode_count': self.episode_count,
                'total_rewards': self.total_rewards
            }, f)
    
    def load(self, filepath: str) -> bool:
        """
        Description:
        ---------------
            Загрузка состояния Q-агента из файла. Восстанавливаются Q-таблица,
            текущая вероятность исследования и статистика.
        
        Args:
        ---------------
            filepath: Путь к файлу для загрузки
        
        Returns:
        ---------------
            bool: True, если загрузка успешна, иначе False
        
        Raises:
        ---------------
            IOError: При проблемах с чтением файла
            pickle.UnpicklingError: При проблемах с десериализацией данных
        
        Examples:
        ---------------
            >>> success = agent.load('q_agent.pkl')
            >>> if success:
            >>>     print("Агент успешно загружен")
        """
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.q_table = data['q_table']
                self.exploration_rate = data['exploration_rate']
                # Используем .get() для совместимости со старыми сохранениями
                self.episode_count = data.get('episode_count', 0)
                self.total_rewards = data.get('total_rewards', [])
                
            return True
        return False