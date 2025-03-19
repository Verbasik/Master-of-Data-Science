# simplenn/core/loss.py
"""
Модуль, содержащий функции потерь для обучения нейронных сетей.
"""

import numpy as np
from typing import Union

class Loss:
    """Базовый класс для функций потерь"""
    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        raise NotImplementedError("Функция потерь должна быть реализована")

    def derivative(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Производная функции потерь должна быть реализована")


class MSE(Loss):
    """Среднеквадратичная ошибка (Mean Squared Error)"""
    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return np.mean(np.square(y_pred - y_true))

    def derivative(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        return 2 * (y_pred - y_true) / y_true.shape[0]


class MAE(Loss):
    """Средняя абсолютная ошибка (Mean Absolute Error)"""
    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return np.mean(np.abs(y_pred - y_true))

    def derivative(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        return np.sign(y_pred - y_true) / y_true.shape[0]


class BinaryCrossEntropy(Loss):
    """Бинарная кросс-энтропия для задач бинарной классификации"""
    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        # Clipping для численной стабильности
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def derivative(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        # Clipping для численной стабильности
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -(y_true / y_pred - (1 - y_true) / (1 - y_pred)) / y_true.shape[0]


class CategoricalCrossEntropy(Loss):
    """Категориальная кросс-энтропия для задач многоклассовой классификации"""
    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        # Clipping для численной стабильности
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        
        # Если y_true в формате one-hot
        if len(y_true.shape) > 1 and y_true.shape[1] > 1:
            return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        else:
            # Если y_true - индексы классов
            batch_size = y_pred.shape[0]
            return -np.mean(np.log(y_pred[np.arange(batch_size), y_true.astype(int)]))

    def derivative(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        # Clipping для численной стабильности
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        
        # Если y_true в формате one-hot
        if len(y_true.shape) > 1 and y_true.shape[1] > 1:
            return -y_true / y_pred / y_true.shape[0]
        else:
            # Если y_true - индексы классов
            batch_size = y_pred.shape[0]
            grad = np.zeros_like(y_pred)
            grad[np.arange(batch_size), y_true.astype(int)] = -1 / y_pred[np.arange(batch_size), y_true.astype(int)]
            return grad / batch_size