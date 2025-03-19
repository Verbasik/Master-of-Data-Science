# simplenn/core/layer.py
"""
Модуль, содержащий реализации слоев нейронной сети.
"""

import numpy as np
from typing import Optional, Dict, Any, Union

from .activation import Activation
from .optimizer import Optimizer

class Layer:
    """Базовый класс слоя нейронной сети"""
    def __init__(self):
        self.params = {}
        self.grads = {}
        self.input = None
        self.output = None

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """Прямой проход через слой"""
        raise NotImplementedError("Метод forward должен быть реализован")

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Обратный проход (расчет градиентов)"""
        raise NotImplementedError("Метод backward должен быть реализован")

    def update_params(self, optimizer: 'Optimizer'):
        """Обновление параметров слоя с помощью оптимизатора"""
        pass


class Dense(Layer):
    """Полносвязный слой нейронной сети"""
    def __init__(
        self, 
        input_size: int, 
        output_size: int, 
        activation: Optional[Activation] = None,
        weight_init: str = 'he',
        use_bias: bool = True
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.use_bias = use_bias
        
        # Инициализация весов в зависимости от метода
        if weight_init == 'xavier':
            # Xavier/Glorot инициализация для сигмоиды и тангенса
            limit = np.sqrt(6 / (input_size + output_size))
            self.params['W'] = np.random.uniform(-limit, limit, (input_size, output_size))
        elif weight_init == 'he':
            # He инициализация для ReLU
            std = np.sqrt(2 / input_size)
            self.params['W'] = np.random.normal(0, std, (input_size, output_size))
        else:
            # Стандартная инициализация
            self.params['W'] = np.random.randn(input_size, output_size) * 0.01
        
        if use_bias:
            self.params['b'] = np.zeros((1, output_size))
        
        self.grads['W'] = np.zeros_like(self.params['W'])
        if use_bias:
            self.grads['b'] = np.zeros_like(self.params['b'])

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        self.input = input_data
        self.output = np.dot(input_data, self.params['W'])
        
        if self.use_bias:
            self.output += self.params['b']
        
        if self.activation:
            self.activated = self.activation(self.output)
            return self.activated
        
        return self.output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        # Если используется функция активации, сначала считаем её градиент
        if self.activation:
            grad_output = grad_output * self.activation.derivative(self.output)
        
        # Вычисляем градиенты по весам и смещению
        self.grads['W'] = np.dot(self.input.T, grad_output)
        
        if self.use_bias:
            self.grads['b'] = np.sum(grad_output, axis=0, keepdims=True)
        
        # Вычисляем градиент для предыдущего слоя
        grad_input = np.dot(grad_output, self.params['W'].T)
        return grad_input

    def update_params(self, optimizer: 'Optimizer'):
        """Обновление параметров слоя с помощью оптимизатора"""
        for param_name in self.params:
            self.params[param_name] = optimizer.update(
                self.params[param_name], 
                self.grads[param_name], 
                param_name
            )


class Dropout(Layer):
    """Слой Dropout для регуляризации нейронной сети"""
    def __init__(self, drop_rate: float = 0.5):
        super().__init__()
        self.drop_rate = drop_rate
        self.mask = None
        self.training = True

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        self.input = input_data
        
        if self.training:
            # Создаем маску для dropout
            self.mask = np.random.binomial(1, 1 - self.drop_rate, size=input_data.shape) / (1 - self.drop_rate)
            # Применяем маску
            self.output = input_data * self.mask
        else:
            # В режиме тестирования не применяем dropout
            self.output = input_data
            
        return self.output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self.training:
            # Применяем ту же маску к градиентам
            return grad_output * self.mask
        return grad_output

    def train(self):
        """Включение режима обучения"""
        self.training = True

    def eval(self):
        """Включение режима тестирования"""
        self.training = False


class BatchNormalization(Layer):
    """Слой батч-нормализации для улучшения сходимости"""
    def __init__(self, input_size: int, momentum: float = 0.9, epsilon: float = 1e-5):
        super().__init__()
        self.input_size = input_size
        self.momentum = momentum
        self.epsilon = epsilon
        
        # Параметры, которые обучаются
        self.params['gamma'] = np.ones((1, input_size))
        self.params['beta'] = np.zeros((1, input_size))
        
        # Градиенты
        self.grads['gamma'] = np.zeros_like(self.params['gamma'])
        self.grads['beta'] = np.zeros_like(self.params['beta'])
        
        # Бегущие средние для статистик (используются при тестировании)
        self.running_mean = np.zeros((1, input_size))
        self.running_var = np.ones((1, input_size))
        
        # Режим работы
        self.training = True

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        self.input = input_data
        
        if self.training:
            # Вычисляем статистики по батчу
            self.batch_mean = np.mean(input_data, axis=0, keepdims=True)
            self.batch_var = np.var(input_data, axis=0, keepdims=True)
            
            # Нормализуем входные данные
            self.x_norm = (input_data - self.batch_mean) / np.sqrt(self.batch_var + self.epsilon)
            
            # Обновляем бегущие средние
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.batch_var
        else:
            # В режиме тестирования используем бегущие средние
            self.x_norm = (input_data - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
        
        # Масштабирование и сдвиг
        self.output = self.params['gamma'] * self.x_norm + self.params['beta']
        return self.output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        # Только во время обучения вычисляем градиенты
        if not self.training:
            return grad_output
        
        batch_size = self.input.shape[0]
        
        # Градиенты для gamma и beta
        self.grads['gamma'] = np.sum(grad_output * self.x_norm, axis=0, keepdims=True)
        self.grads['beta'] = np.sum(grad_output, axis=0, keepdims=True)
        
        # Градиент по нормализованному входу
        dx_norm = grad_output * self.params['gamma']
        
        # Градиент по дисперсии
        dvar = np.sum(dx_norm * (self.input - self.batch_mean) * (-0.5) * 
                      np.power(self.batch_var + self.epsilon, -1.5), axis=0, keepdims=True)
        
        # Градиент по среднему значению
        dmean = np.sum(dx_norm * (-1) / np.sqrt(self.batch_var + self.epsilon), axis=0, keepdims=True) + \
                dvar * np.mean(-2 * (self.input - self.batch_mean), axis=0, keepdims=True)
        
        # Градиент по входу
        grad_input = dx_norm / np.sqrt(self.batch_var + self.epsilon) + \
                     dvar * 2 * (self.input - self.batch_mean) / batch_size + \
                     dmean / batch_size
        
        return grad_input

    def train(self):
        """Включение режима обучения"""
        self.training = True

    def eval(self):
        """Включение режима тестирования"""
        self.training = False

    def update_params(self, optimizer: 'Optimizer'):
        """Обновление параметров слоя с помощью оптимизатора"""
        for param_name in ['gamma', 'beta']:
            self.params[param_name] = optimizer.update(
                self.params[param_name], 
                self.grads[param_name], 
                param_name
            )