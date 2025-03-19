# simplenn/core/activation.py
"""
Модуль, содержащий функции активации для нейронных сетей.
"""

import numpy as np

class Activation:
    """Базовый класс для функций активации"""
    def __call__(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Функция активации должна быть реализована")

    def derivative(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Производная функции активации должна быть реализована")


class Sigmoid(Activation):
    """Сигмоидная функция активации"""
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

    def derivative(self, x: np.ndarray) -> np.ndarray:
        s = self.__call__(x)
        return s * (1 - s)


class ReLU(Activation):
    """Функция активации ReLU"""
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1, 0)


class LeakyReLU(Activation):
    """Функция активации Leaky ReLU"""
    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(self.alpha * x, x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1, self.alpha)


class Tanh(Activation):
    """Гиперболический тангенс"""
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return 1 - np.tanh(x) ** 2


class Softmax(Activation):
    """Функция активации Softmax для многоклассовой классификации"""
    def __call__(self, x: np.ndarray) -> np.ndarray:
        # Нормализация для численной стабильности
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        # Производная Softmax для кросс-энтропии обычно не требуется, 
        # так как она упрощается в сочетании с функцией потерь
        # Возвращаем единичную матрицу для совместимости
        return np.ones_like(x)