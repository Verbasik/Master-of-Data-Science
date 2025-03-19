# simplenn/core/tensor.py
"""
Модуль для работы с тензорами - основными структурами данных для нейронных сетей.
"""

import numpy as np

class Tensor:
    """
    Базовый класс для работы с тензорами
    """
    def __init__(self, data: np.ndarray):
        self.data = data
        self.grad = np.zeros_like(data, dtype=np.float64)

    def zero_grad(self):
        """Обнуление градиентов"""
        self.grad = np.zeros_like(self.data, dtype=np.float64)

    @property
    def shape(self):
        return self.data.shape