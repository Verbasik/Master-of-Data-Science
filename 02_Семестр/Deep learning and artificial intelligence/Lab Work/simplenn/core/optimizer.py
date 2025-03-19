# simplenn/core/optimizer.py
"""
Модуль, содержащий реализации оптимизаторов для обучения нейронных сетей.
"""

import numpy as np
from typing import Dict

class Optimizer:
    """Базовый класс оптимизатора"""
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate

    def update(
        self, 
        param: np.ndarray, 
        grad: np.ndarray, 
        param_name: str
    ) -> np.ndarray:
        """Обновление параметров"""
        raise NotImplementedError("Метод update должен быть реализован")


class SGD(Optimizer):
    """Стохастический градиентный спуск (Stochastic Gradient Descent)"""
    def update(
        self, 
        param: np.ndarray, 
        grad: np.ndarray, 
        param_name: str
    ) -> np.ndarray:
        """Обновление параметров с помощью SGD"""
        return param - self.learning_rate * grad


class MomentumSGD(Optimizer):
    """SGD с моментом для ускорения сходимости"""
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocities = {}  # словарь для хранения скоростей

    def update(
        self, 
        param: np.ndarray, 
        grad: np.ndarray, 
        param_name: str
    ) -> np.ndarray:
        """Обновление параметров с моментом"""
        # Инициализируем скорость, если её еще нет
        if param_name not in self.velocities:
            self.velocities[param_name] = np.zeros_like(param)
        
        # Обновляем скорость
        self.velocities[param_name] = (
            self.momentum * self.velocities[param_name] - 
            self.learning_rate * grad
        )
        
        # Обновляем параметр
        return param + self.velocities[param_name]


class Adagrad(Optimizer):
    """Адаптивный градиентный алгоритм"""
    def __init__(self, learning_rate: float = 0.01, epsilon: float = 1e-8):
        super().__init__(learning_rate)
        self.epsilon = epsilon
        self.cache = {}  # накопленные квадраты градиентов

    def update(
        self, 
        param: np.ndarray, 
        grad: np.ndarray, 
        param_name: str
    ) -> np.ndarray:
        """Обновление параметров с помощью Adagrad"""
        # Инициализируем кэш, если его еще нет
        if param_name not in self.cache:
            self.cache[param_name] = np.zeros_like(param)
        
        # Накапливаем квадраты градиентов
        self.cache[param_name] += np.square(grad)
        
        # Обновляем параметр с адаптивной скоростью обучения
        return param - self.learning_rate * grad / (np.sqrt(self.cache[param_name]) + self.epsilon)


class RMSprop(Optimizer):
    """Оптимизатор RMSprop (Root Mean Square Propagation)"""
    def __init__(self, learning_rate: float = 0.01, decay_rate: float = 0.9, epsilon: float = 1e-8):
        super().__init__(learning_rate)
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.cache = {}  # Экспоненциально взвешенное среднее квадратов градиентов

    def update(
        self, 
        param: np.ndarray, 
        grad: np.ndarray, 
        param_name: str
    ) -> np.ndarray:
        """Обновление параметров с помощью RMSprop"""
        # Инициализируем кэш, если его еще нет
        if param_name not in self.cache:
            self.cache[param_name] = np.zeros_like(param)
        
        # Обновляем экспоненциально взвешенное среднее
        self.cache[param_name] = (
            self.decay_rate * self.cache[param_name] + 
            (1 - self.decay_rate) * np.square(grad)
        )
        
        # Обновляем параметр
        return param - self.learning_rate * grad / (np.sqrt(self.cache[param_name]) + self.epsilon)


class Adam(Optimizer):
    """Адаптивная оценка момента (Adaptive Moment Estimation)"""
    def __init__(
        self, 
        learning_rate: float = 0.001, 
        beta1: float = 0.9, 
        beta2: float = 0.999, 
        epsilon: float = 1e-8
    ):
        super().__init__(learning_rate)
        self.beta1 = beta1  # коэффициент для момента первого порядка
        self.beta2 = beta2  # коэффициент для момента второго порядка
        self.epsilon = epsilon
        self.m = {}  # момент первого порядка
        self.v = {}  # момент второго порядка
        self.t = 0  # счетчик шагов

    def update(
        self, 
        param: np.ndarray, 
        grad: np.ndarray, 
        param_name: str
    ) -> np.ndarray:
        """Обновление параметров с помощью Adam"""
        # Увеличиваем счетчик шагов
        self.t += 1
        
        # Инициализируем моменты, если их еще нет
        if param_name not in self.m:
            self.m[param_name] = np.zeros_like(param)
            self.v[param_name] = np.zeros_like(param)
        
        # Обновляем моменты
        self.m[param_name] = self.beta1 * self.m[param_name] + (1 - self.beta1) * grad
        self.v[param_name] = self.beta2 * self.v[param_name] + (1 - self.beta2) * np.square(grad)
        
        # Корректируем моменты (коррекция смещения)
        m_corrected = self.m[param_name] / (1 - self.beta1 ** self.t)
        v_corrected = self.v[param_name] / (1 - self.beta2 ** self.t)
        
        # Обновляем параметр
        return param - self.learning_rate * m_corrected / (np.sqrt(v_corrected) + self.epsilon)


class GradientClipping(Optimizer):
    """
    Оптимизатор с отсечением градиентов для предотвращения проблемы взрывающихся градиентов
    Работает как обертка над другим оптимизатором
    """
    def __init__(self, optimizer: Optimizer, clip_value: float = 5.0):
        self.optimizer = optimizer
        self.clip_value = clip_value
        self.learning_rate = optimizer.learning_rate

    def update(
        self, 
        param: np.ndarray, 
        grad: np.ndarray, 
        param_name: str
    ) -> np.ndarray:
        """Отсечение градиентов и обновление параметров"""
        # Отсекаем градиенты по норме
        grad_norm = np.linalg.norm(grad)
        if grad_norm > self.clip_value:
            grad = grad * self.clip_value / grad_norm
        
        # Используем базовый оптимизатор для обновления параметров
        return self.optimizer.update(param, grad, param_name)