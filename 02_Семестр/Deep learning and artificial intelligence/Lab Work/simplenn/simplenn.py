# Стандартные библиотеки
import os
import gzip
import pickle
from typing import List, Dict, Any, Tuple, Optional, Union, Callable, Type

# Библиотеки для работы с данными
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Библиотеки для визуализации
import matplotlib.pyplot as plt
from tqdm import tqdm


class Activation:
    """
    Description:
    ---------------
        Базовый класс для функций активации в нейронной сети.
        Определяет интерфейс для всех функций активации.
    
    Args:
    ---------------
        Нет аргументов при инициализации.
    
    Returns:
    ---------------
        Нет возвращаемых значений.
    
    Raises:
    ---------------
        NotImplementedError: Вызывается, если дочерний класс не реализует
                            необходимые методы.
    
    Examples:
    ---------------
        >>> # Создание производного класса
        >>> class CustomActivation(Activation):
        ...     def forward(self, x):
        ...         return x
        ...     def backward(self, x):
        ...         return np.ones_like(x)
    """
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Description:
        ---------------
            Выполняет прямой проход через функцию активации.
        
        Args:
        ---------------
            x: Входные данные для функции активации.
        
        Returns:
        ---------------
            Активированные выходные данные.
        
        Raises:
        ---------------
            NotImplementedError: Должен быть реализован в дочернем классе.
        """
        raise NotImplementedError
        
    def backward(self, x: np.ndarray) -> np.ndarray:
        """
        Description:
        ---------------
            Выполняет обратный проход через функцию активации 
            (расчет производной).
        
        Args:
        ---------------
            x: Входные данные для расчета производной.
        
        Returns:
        ---------------
            Градиент функции активации.
        
        Raises:
        ---------------
            NotImplementedError: Должен быть реализован в дочернем классе.
        """
        raise NotImplementedError


class Sigmoid(Activation):
    """
    Description:
    ---------------
        Сигмоидная функция активации: f(x) = 1 / (1 + exp(-x)).
        Используется в задачах бинарной классификации.
    
    Args:
    ---------------
        Нет аргументов при инициализации.
    
    Returns:
    ---------------
        Нет возвращаемых значений.
    
    Examples:
    ---------------
        >>> sigmoid = Sigmoid()
        >>> sigmoid.forward(np.array([0]))
        array([0.5])
        >>> sigmoid.backward(np.array([0]))
        array([0.25])
    """
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Description:
        ---------------
            Выполняет прямое распространение сигмоидной функции.
            С ограничением значений для численной стабильности.
        
        Args:
        ---------------
            x: Входные данные для функции активации.
        
        Returns:
        ---------------
            Активированные выходные данные (значения в диапазоне [0, 1]).
        """
        return 1 / (1 + np.exp(-np.clip(x, -15, 15)))
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        """
        Description:
        ---------------
            Расчет производной сигмоидной функции: f'(x) = f(x) * (1 - f(x)).
        
        Args:
        ---------------
            x: Входные данные для расчета производной.
        
        Returns:
        ---------------
            Градиент сигмоидной функции.
        """
        s = self.forward(x)
        return s * (1 - s)


class ReLU(Activation):
    """
    Description:
    ---------------
        Функция активации ReLU (Rectified Linear Unit): f(x) = max(0, x).
        Часто используется в скрытых слоях нейронных сетей.
    
    Args:
    ---------------
        Нет аргументов при инициализации.
    
    Returns:
    ---------------
        Нет возвращаемых значений.
    
    Examples:
    ---------------
        >>> relu = ReLU()
        >>> relu.forward(np.array([-1, 0, 1]))
        array([0, 0, 1])
        >>> relu.backward(np.array([-1, 0, 1]))
        array([0, 0, 1])
    """
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Description:
        ---------------
            Выполняет прямое распространение ReLU функции.
        
        Args:
        ---------------
            x: Входные данные для функции активации.
        
        Returns:
        ---------------
            Активированные выходные данные (x, если x > 0, иначе 0).
        """
        return np.maximum(0, x)
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        """
        Description:
        ---------------
            Расчет производной ReLU функции: f'(x) = 1, если x > 0, иначе 0.
        
        Args:
        ---------------
            x: Входные данные для расчета производной.
        
        Returns:
        ---------------
            Градиент ReLU функции.
        """
        return np.where(x > 0, 1, 0)


class Tanh(Activation):
    """
    Description:
    ---------------
        Функция активации гиперболический тангенс: f(x) = tanh(x).
        Возвращает значения в диапазоне [-1, 1].
    
    Args:
    ---------------
        Нет аргументов при инициализации.
    
    Returns:
    ---------------
        Нет возвращаемых значений.
    
    Examples:
    ---------------
        >>> tanh = Tanh()
        >>> tanh.forward(np.array([0]))
        array([0.])
        >>> tanh.backward(np.array([0]))
        array([1.])
    """
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Description:
        ---------------
            Выполняет прямое распространение функции tanh.
        
        Args:
        ---------------
            x: Входные данные для функции активации.
        
        Returns:
        ---------------
            Активированные выходные данные (значения в диапазоне [-1, 1]).
        """
        return np.tanh(x)
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        """
        Description:
        ---------------
            Расчет производной функции tanh: f'(x) = 1 - tanh(x)^2.
        
        Args:
        ---------------
            x: Входные данные для расчета производной.
        
        Returns:
        ---------------
            Градиент функции tanh.
        """
        return 1 - np.tanh(x)**2


class Softmax(Activation):
    """
    Description:
    ---------------
        Функция активации Softmax для многоклассовой классификации.
        Преобразует вектор чисел в распределение вероятностей.
    
    Args:
    ---------------
        Нет аргументов при инициализации.
    
    Returns:
    ---------------
        Нет возвращаемых значений.
    
    Examples:
    ---------------
        >>> softmax = Softmax()
        >>> softmax.forward(np.array([[1, 2, 3]]))
        array([[0.09003057, 0.24472847, 0.66524096]])
    """
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Description:
        ---------------
            Выполняет прямое распространение функции Softmax.
            Для численной стабильности вычитается максимальное значение.
        
        Args:
        ---------------
            x: Входные данные для функции активации.
        
        Returns:
        ---------------
            Активированные выходные данные (вероятности, сумма = 1).
        """
        # Для численной стабильности вычитаем максимальное значение
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        """
        Description:
        ---------------
            Расчет производной Softmax. В комбинации с CrossEntropy 
            градиент обычно упрощается.
        
        Args:
        ---------------
            x: Входные данные для расчета производной.
        
        Returns:
        ---------------
            Градиент функции Softmax (упрощенная версия для 
            использования с CrossEntropy).
        """
        # В комбинации с CrossEntropy градиент обычно упрощается
        # Возвращаем единичную матрицу, т.к. градиент уже учтён 
        # в функции потерь
        return np.ones_like(x)


class Layer:
    """
    Description:
    ---------------
        Базовый класс для слоев нейронной сети.
        Определяет интерфейс для всех слоев.
    
    Args:
    ---------------
        Нет аргументов при инициализации.
    
    Returns:
    ---------------
        Нет возвращаемых значений.
    
    Raises:
    ---------------
        NotImplementedError: Вызывается, если дочерний класс не реализует
                            необходимые методы.
    """
    
    def __init__(self) -> None:
        """
        Description:
        ---------------
            Инициализирует базовый слой.
        """
        self.params = {}
        self.grads = {}
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Description:
        ---------------
            Выполняет прямой проход через слой.
        
        Args:
        ---------------
            x: Входные данные для слоя.
        
        Returns:
        ---------------
            Выходные данные слоя.
        
        Raises:
        ---------------
            NotImplementedError: Должен быть реализован в дочернем классе.
        """
        raise NotImplementedError
        
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """
        Description:
        ---------------
            Выполняет обратный проход через слой (расчет градиентов).
        
        Args:
        ---------------
            grad: Градиент от следующего слоя.
        
        Returns:
        ---------------
            Градиент для предыдущего слоя.
        
        Raises:
        ---------------
            NotImplementedError: Должен быть реализован в дочернем классе.
        """
        raise NotImplementedError


class Dense(Layer):
    """
    Description:
    ---------------
        Полносвязный слой нейронной сети.
        Выполняет операцию y = activation(x * W + b).
    
    Args:
    ---------------
        input_size: Размер входного вектора.
        output_size: Размер выходного вектора.
        activation: Функция активации (опционально).
    
    Returns:
    ---------------
        Нет возвращаемых значений.
    
    Examples:
    ---------------
        >>> layer = Dense(3, 2, activation=ReLU())
        >>> layer.forward(np.array([[1, 2, 3]]))
        array([[...]])  # Результат зависит от инициализации весов
    """
    
    def __init__(
        self, 
        input_size: int, 
        output_size: int, 
        activation: Optional[Activation] = None
    ) -> None:
        """
        Description:
        ---------------
            Инициализирует полносвязный слой.
        
        Args:
        ---------------
            input_size: Размер входного вектора.
            output_size: Размер выходного вектора.
            activation: Функция активации (опционально).
        """
        super().__init__()
        
        # Инициализация весов с помощью метода Xavier/Glorot
        self.params['W'] = np.random.randn(input_size, output_size) * np.sqrt(
            2 / (input_size + output_size)
        )
        self.params['b'] = np.zeros(output_size)
        
        self.activation = activation
        self.x = None
        self.z = None
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Description:
        ---------------
            Выполняет прямой проход через полносвязный слой.
        
        Args:
        ---------------
            x: Входные данные для слоя.
        
        Returns:
        ---------------
            Выходные данные слоя.
        """
        self.x = x
        self.z = np.dot(x, self.params['W']) + self.params['b']
        
        if self.activation:
            return self.activation.forward(self.z)
        else:
            return self.z
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """
        Description:
        ---------------
            Выполняет обратный проход через полносвязный слой.
        
        Args:
        ---------------
            grad: Градиент от следующего слоя.
        
        Returns:
        ---------------
            Градиент для предыдущего слоя.
        """
        if self.activation:
            grad = grad * self.activation.backward(self.z)
            
        self.grads['W'] = np.dot(self.x.T, grad)
        self.grads['b'] = np.sum(grad, axis=0)
        
        return np.dot(grad, self.params['W'].T)


class Dropout(Layer):
    """
    Description:
    ---------------
        Слой Dropout для регуляризации.
        Случайно отключает нейроны во время обучения для 
        предотвращения переобучения.
    
    Args:
    ---------------
        drop_rate: Вероятность отключения нейрона (от 0 до 1).
    
    Returns:
    ---------------
        Нет возвращаемых значений.
    
    Examples:
    ---------------
        >>> layer = Dropout(drop_rate=0.5)
        >>> x = np.ones((2, 3))
        >>> layer.training = True
        >>> output = layer.forward(x)  # Случайно маскирует входы
        >>> layer.training = False
        >>> output = layer.forward(x)  # Равно x во время тестирования
    """
    
    def __init__(self, drop_rate: float = 0.5) -> None:
        """
        Description:
        ---------------
            Инициализирует слой Dropout.
        
        Args:
        ---------------
            drop_rate: Вероятность отключения нейрона (от 0 до 1).
        """
        super().__init__()
        self.drop_rate = drop_rate
        self.mask = None
        self.training = True
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Description:
        ---------------
            Выполняет прямой проход через слой Dropout.
            Во время обучения случайно отключает нейроны.
            Во время тестирования просто передает данные дальше.
        
        Args:
        ---------------
            x: Входные данные для слоя.
        
        Returns:
        ---------------
            Выходные данные с примененным Dropout (во время обучения)
            или исходные данные (во время тестирования).
        """
        if self.training:
            self.mask = np.random.binomial(
                1, 
                1-self.drop_rate, 
                size=x.shape
            ) / (1-self.drop_rate)
            return x * self.mask
        else:
            return x
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """
        Description:
        ---------------
            Выполняет обратный проход через слой Dropout.
        
        Args:
        ---------------
            grad: Градиент от следующего слоя.
        
        Returns:
        ---------------
            Градиент для предыдущего слоя.
        """
        if self.training:
            return grad * self.mask
        else:
            return grad


class BatchNormalization(Layer):
    """
    Description:
    ---------------
        Слой батч-нормализации для стабилизации и ускорения обучения.
        Нормализует входы слоя по мини-батчу.
    
    Args:
    ---------------
        input_dim: Размерность входных данных.
        epsilon: Малая константа для численной стабильности.
        momentum: Параметр для экспоненциального среднего статистик.
    
    Returns:
    ---------------
        Нет возвращаемых значений.
    
    Examples:
    ---------------
        >>> layer = BatchNormalization(3)
        >>> x = np.random.randn(2, 3)
        >>> layer.forward(x)  # Нормализованные выходы
    """
    
    def __init__(
        self, 
        input_dim: int, 
        epsilon: float = 1e-8, 
        momentum: float = 0.9
    ) -> None:
        """
        Description:
        ---------------
            Инициализирует слой батч-нормализации.
        
        Args:
        ---------------
            input_dim: Размерность входных данных.
            epsilon: Малая константа для численной стабильности.
            momentum: Параметр для экспоненциального среднего статистик.
        """
        super().__init__()
        self.epsilon = epsilon
        self.momentum = momentum
        
        # Обучаемые параметры
        self.params['gamma'] = np.ones(input_dim)
        self.params['beta'] = np.zeros(input_dim)
        
        # Параметры для вычислений
        self.running_mean = np.zeros(input_dim)
        self.running_var = np.ones(input_dim)
        
        self.x = None
        self.x_norm = None
        self.batch_mean = None
        self.batch_var = None
        self.training = True
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Description:
        ---------------
            Выполняет прямой проход через слой батч-нормализации.
        
        Args:
        ---------------
            x: Входные данные для слоя.
        
        Returns:
        ---------------
            Нормализованные выходные данные.
        """
        self.x = x
        
        if self.training:
            self.batch_mean = np.mean(x, axis=0)
            self.batch_var = np.var(x, axis=0)
            
            # Обновляем running статистики
            self.running_mean = (
                self.momentum * self.running_mean + 
                (1 - self.momentum) * self.batch_mean
            )
            self.running_var = (
                self.momentum * self.running_var + 
                (1 - self.momentum) * self.batch_var
            )
            
            # Нормализуем
            self.x_norm = (x - self.batch_mean) / np.sqrt(
                self.batch_var + self.epsilon
            )
            
        else:
            # В режиме вывода используем running статистики
            self.x_norm = (x - self.running_mean) / np.sqrt(
                self.running_var + self.epsilon
            )
        
        # Применяем масштабирование и сдвиг
        return self.params['gamma'] * self.x_norm + self.params['beta']
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """
        Description:
        ---------------
            Выполняет обратный проход через слой батч-нормализации.
        
        Args:
        ---------------
            grad: Градиент от следующего слоя.
        
        Returns:
        ---------------
            Градиент для предыдущего слоя.
        """
        # Вычисляем градиенты для gamma и beta
        self.grads['gamma'] = np.sum(grad * self.x_norm, axis=0)
        self.grads['beta'] = np.sum(grad, axis=0)
        
        # Вычисляем градиент для входа
        N = self.x.shape[0]
        dx_norm = grad * self.params['gamma']
        
        # Градиент для батч-нормализации
        dx = (1. / N) * (1. / np.sqrt(self.batch_var + self.epsilon)) * (
            N * dx_norm - np.sum(dx_norm, axis=0) - 
            self.x_norm * np.sum(dx_norm * self.x_norm, axis=0)
        )
        
        return dx


class Loss:
    """
    Description:
    ---------------
        Базовый класс для функций потерь.
        Определяет интерфейс для всех функций потерь.
    
    Args:
    ---------------
        Нет аргументов при инициализации.
    
    Returns:
    ---------------
        Нет возвращаемых значений.
    
    Raises:
    ---------------
        NotImplementedError: Вызывается, если дочерний класс не реализует
                            необходимые методы.
    """
    
    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Description:
        ---------------
            Переопределение оператора вызова для удобства использования.
        
        Args:
        ---------------
            y_pred: Предсказанные значения.
            y_true: Истинные значения.
        
        Returns:
        ---------------
            Значение функции потерь.
        """
        return self.forward(y_pred, y_true)
    
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Description:
        ---------------
            Выполняет прямой расчет функции потерь.
        
        Args:
        ---------------
            y_pred: Предсказанные значения.
            y_true: Истинные значения.
        
        Returns:
        ---------------
            Значение функции потерь.
        
        Raises:
        ---------------
            NotImplementedError: Должен быть реализован в дочернем классе.
        """
        raise NotImplementedError
        
    def backward(
        self, y_pred: np.ndarray, y_true: np.ndarray
    ) -> np.ndarray:
        """
        Description:
        ---------------
            Выполняет обратный расчет градиента функции потерь.
        
        Args:
        ---------------
            y_pred: Предсказанные значения.
            y_true: Истинные значения.
        
        Returns:
        ---------------
            Градиент функции потерь по предсказанным значениям.
        
        Raises:
        ---------------
            NotImplementedError: Должен быть реализован в дочернем классе.
        """
        raise NotImplementedError


class MSE(Loss):
    """
    Description:
    ---------------
        Среднеквадратичная ошибка (Mean Squared Error).
        Используется для задач регрессии.
    
    Args:
    ---------------
        Нет аргументов при инициализации.
    
    Returns:
    ---------------
        Нет возвращаемых значений.
    
    Examples:
    ---------------
        >>> loss = MSE()
        >>> y_true = np.array([[1, 2]])
        >>> y_pred = np.array([[1.1, 1.9]])
        >>> loss(y_pred, y_true)
        0.01
    """
    
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Description:
        ---------------
            Расчет среднеквадратичной ошибки.
        
        Args:
        ---------------
            y_pred: Предсказанные значения.
            y_true: Истинные значения.
        
        Returns:
        ---------------
            Значение MSE.
        """
        return np.mean(np.square(y_pred - y_true))
    
    def backward(
        self, y_pred: np.ndarray, y_true: np.ndarray
    ) -> np.ndarray:
        """
        Description:
        ---------------
            Расчет градиента MSE.
        
        Args:
        ---------------
            y_pred: Предсказанные значения.
            y_true: Истинные значения.
        
        Returns:
        ---------------
            Градиент MSE по предсказанным значениям.
        """
        return 2 * (y_pred - y_true) / y_pred.shape[0]


class CrossEntropy(Loss):
    """
    Description:
    ---------------
        Категориальная кросс-энтропия с Softmax.
        Используется для задач многоклассовой классификации.
    
    Args:
    ---------------
        Нет аргументов при инициализации.
    
    Returns:
    ---------------
        Нет возвращаемых значений.
    
    Examples:
    ---------------
        >>> loss = CrossEntropy()
        >>> y_true = np.array([[0, 1, 0]])
        >>> y_pred = np.array([[0.1, 0.8, 0.1]])
        >>> loss(y_pred, y_true)
        0.2231435513142097
    """
    
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Description:
        ---------------
            Расчет категориальной кросс-энтропии.
        
        Args:
        ---------------
            y_pred: Предсказанные вероятности.
            y_true: Истинные метки в формате one-hot.
        
        Returns:
        ---------------
            Значение кросс-энтропии.
        """
        # Для численной стабильности
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    
    def backward(
        self, y_pred: np.ndarray, y_true: np.ndarray
    ) -> np.ndarray:
        """
        Description:
        ---------------
            Расчет градиента кросс-энтропии с Softmax.
        
        Args:
        ---------------
            y_pred: Предсказанные вероятности.
            y_true: Истинные метки в формате one-hot.
        
        Returns:
        ---------------
            Градиент кросс-энтропии по предсказанным значениям.
        """
        # Прямой градиент для комбинации Softmax + CrossEntropy
        return (y_pred - y_true) / y_pred.shape[0]


class BinaryCrossEntropy(Loss):
    """
    Description:
    ---------------
        Бинарная кросс-энтропия.
        Используется для задач бинарной классификации.
    
    Args:
    ---------------
        Нет аргументов при инициализации.
    
    Returns:
    ---------------
        Нет возвращаемых значений.
    
    Examples:
    ---------------
        >>> loss = BinaryCrossEntropy()
        >>> y_true = np.array([[1]])
        >>> y_pred = np.array([[0.8]])
        >>> loss(y_pred, y_true)
        0.2231435513142097
    """
    
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Description:
        ---------------
            Расчет бинарной кросс-энтропии.
        
        Args:
        ---------------
            y_pred: Предсказанные вероятности.
            y_true: Истинные метки (0 или 1).
        
        Returns:
        ---------------
            Значение бинарной кросс-энтропии.
        """
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(
            y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
        )
    
def backward(
        self, y_pred: np.ndarray, y_true: np.ndarray
    ) -> np.ndarray:
        """
        Description:
        ---------------
            Расчет градиента бинарной кросс-энтропии.
        
        Args:
        ---------------
            y_pred: Предсказанные вероятности.
            y_true: Истинные метки (0 или 1).
        
        Returns:
        ---------------
            Градиент бинарной кросс-энтропии по предсказанным значениям.
        """
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -(y_true / y_pred - (1 - y_true) / (1 - y_pred)) / y_pred.shape[0]


class Optimizer:
    """
    Description:
    ---------------
        Базовый класс для оптимизаторов.
        Определяет интерфейс для всех оптимизаторов.
    
    Args:
    ---------------
        lr: Скорость обучения (learning rate).
    
    Returns:
    ---------------
        Нет возвращаемых значений.
    
    Raises:
    ---------------
        NotImplementedError: Вызывается, если дочерний класс не реализует
                            необходимые методы.
    """
    
    def __init__(self, lr: float = 0.01) -> None:
        """
        Description:
        ---------------
            Инициализирует базовый оптимизатор.
        
        Args:
        ---------------
            lr: Скорость обучения (learning rate).
        """
        self.lr = lr
    
    def update(self, layer: Layer) -> None:
        """
        Description:
        ---------------
            Обновляет параметры слоя на основе градиентов.
        
        Args:
        ---------------
            layer: Слой, параметры которого нужно обновить.
        
        Raises:
        ---------------
            NotImplementedError: Должен быть реализован в дочернем классе.
        """
        raise NotImplementedError


class SGD(Optimizer):
    """
    Description:
    ---------------
        Стохастический градиентный спуск.
        Самый простой алгоритм оптимизации.
    
    Args:
    ---------------
        lr: Скорость обучения (learning rate).
    
    Returns:
    ---------------
        Нет возвращаемых значений.
    
    Examples:
    ---------------
        >>> optimizer = SGD(lr=0.01)
        >>> # Предполагается, что слой уже имеет параметры и градиенты
        >>> optimizer.update(layer)
    """
    
    def update(self, layer: Layer) -> None:
        """
        Description:
        ---------------
            Обновляет параметры слоя с помощью SGD.
        
        Args:
        ---------------
            layer: Слой, параметры которого нужно обновить.
        """
        for param_name in layer.params:
            layer.params[param_name] -= self.lr * layer.grads[param_name]


class MomentumSGD(Optimizer):
    """
    Description:
    ---------------
        Стохастический градиентный спуск с моментом.
        Добавляет инерцию к обновлениям для лучшей сходимости.
    
    Args:
    ---------------
        lr: Скорость обучения (learning rate).
        momentum: Коэффициент момента (от 0 до 1).
    
    Returns:
    ---------------
        Нет возвращаемых значений.
    
    Examples:
    ---------------
        >>> optimizer = MomentumSGD(lr=0.01, momentum=0.9)
        >>> # Предполагается, что слой уже имеет параметры и градиенты
        >>> optimizer.update(layer)
    """
    
    def __init__(self, lr: float = 0.01, momentum: float = 0.9) -> None:
        """
        Description:
        ---------------
            Инициализирует оптимизатор SGD с моментом.
        
        Args:
        ---------------
            lr: Скорость обучения (learning rate).
            momentum: Коэффициент момента (от 0 до 1).
        """
        super().__init__(lr)
        self.momentum = momentum
        self.velocities = {}
    
    def update(self, layer: Layer) -> None:
        """
        Description:
        ---------------
            Обновляет параметры слоя с помощью SGD с моментом.
        
        Args:
        ---------------
            layer: Слой, параметры которого нужно обновить.
        """
        # Создаем уникальный ключ для каждого слоя
        layer_id = id(layer)
        
        if layer_id not in self.velocities:
            self.velocities[layer_id] = {}
        
        # Инициализируем скорости, если еще не инициализированы
        for param_name in layer.params:
            if param_name not in self.velocities[layer_id]:
                self.velocities[layer_id][param_name] = np.zeros_like(
                    layer.params[param_name]
                )
            
            # Обновляем скорости и параметры
            self.velocities[layer_id][param_name] = (
                self.momentum * self.velocities[layer_id][param_name] - 
                self.lr * layer.grads[param_name]
            )
            layer.params[param_name] += self.velocities[layer_id][param_name]


class Adam(Optimizer):
    """
    Description:
    ---------------
        Оптимизатор Adam (Adaptive Moment Estimation).
        Адаптивный алгоритм оптимизации, сочетающий преимущества 
        RMSProp и Momentum.
    
    Args:
    ---------------
        lr: Скорость обучения (learning rate).
        beta1: Коэффициент экспоненциального сглаживания для момента.
        beta2: Коэффициент экспоненциального сглаживания для квадрата градиента.
        epsilon: Малая константа для численной стабильности.
    
    Returns:
    ---------------
        Нет возвращаемых значений.
    
    Examples:
    ---------------
        >>> optimizer = Adam(lr=0.001)
        >>> # Предполагается, что слой уже имеет параметры и градиенты
        >>> optimizer.update(layer)
    """
    
    def __init__(
        self, 
        lr: float = 0.001, 
        beta1: float = 0.9, 
        beta2: float = 0.999, 
        epsilon: float = 1e-8
    ) -> None:
        """
        Description:
        ---------------
            Инициализирует оптимизатор Adam.
        
        Args:
        ---------------
            lr: Скорость обучения (learning rate).
            beta1: Коэффициент экспоненциального сглаживания для момента.
            beta2: Коэффициент экспоненциального сглаживания для 
                  квадрата градиента.
            epsilon: Малая константа для численной стабильности.
        """
        super().__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0
    
    def update(self, layer: Layer) -> None:
        """
        Description:
        ---------------
            Обновляет параметры слоя с помощью Adam.
        
        Args:
        ---------------
            layer: Слой, параметры которого нужно обновить.
        """
        self.t += 1
        
        # Создаем уникальный ключ для каждого слоя
        layer_id = id(layer)
        
        if layer_id not in self.m:
            self.m[layer_id] = {}
            self.v[layer_id] = {}
        
        for param_name in layer.params:
            # Инициализируем моменты, если еще не инициализированы
            if param_name not in self.m[layer_id]:
                self.m[layer_id][param_name] = np.zeros_like(
                    layer.params[param_name]
                )
                self.v[layer_id][param_name] = np.zeros_like(
                    layer.params[param_name]
                )
            
            # Обновляем средние значения градиентов и их квадратов
            self.m[layer_id][param_name] = (
                self.beta1 * self.m[layer_id][param_name] + 
                (1 - self.beta1) * layer.grads[param_name]
            )
            self.v[layer_id][param_name] = (
                self.beta2 * self.v[layer_id][param_name] + 
                (1 - self.beta2) * np.square(layer.grads[param_name])
            )
            
            # Корректируем смещение
            m_corrected = self.m[layer_id][param_name] / (
                1 - self.beta1 ** self.t
            )
            v_corrected = self.v[layer_id][param_name] / (
                1 - self.beta2 ** self.t
            )
            
            # Обновляем параметры
            layer.params[param_name] -= (
                self.lr * m_corrected / (
                    np.sqrt(v_corrected) + self.epsilon
                )
            )


class DataLoader:
    """
    Description:
    ---------------
        Класс для загрузки и обработки данных в мини-батчах.
        Позволяет эффективно итерироваться по данным во время обучения.
    
    Args:
    ---------------
        X: Входные данные.
        y: Целевые значения.
        batch_size: Размер мини-батча.
        shuffle: Флаг перемешивания данных перед каждой эпохой.
    
    Returns:
    ---------------
        Нет возвращаемых значений.
    
    Examples:
    ---------------
        >>> X = np.random.randn(100, 10)
        >>> y = np.random.randn(100, 2)
        >>> data_loader = DataLoader(X, y, batch_size=32)
        >>> for X_batch, y_batch in data_loader:
        ...     # Обработка одного батча
        ...     pass
    """
    
    def __init__(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        batch_size: int = 32, 
        shuffle: bool = True
    ) -> None:
        """
        Description:
        ---------------
            Инициализирует загрузчик данных.
        
        Args:
        ---------------
            X: Входные данные.
            y: Целевые значения.
            batch_size: Размер мини-батча.
            shuffle: Флаг перемешивания данных перед каждой эпохой.
        """
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = X.shape[0]
        self.n_batches = int(np.ceil(self.n_samples / self.batch_size))
        self.idx = np.arange(self.n_samples)
    
    def __iter__(self) -> 'DataLoader':
        """
        Description:
        ---------------
            Реализует протокол итератора для загрузчика данных.
            При необходимости перемешивает данные.
        
        Returns:
        ---------------
            Итератор по мини-батчам.
        """
        if self.shuffle:
            np.random.shuffle(self.idx)
        
        for i in range(self.n_batches):
            start_idx = i * self.batch_size
            end_idx = min(start_idx + self.batch_size, self.n_samples)
            batch_idx = self.idx[start_idx:end_idx]
            
            yield self.X[batch_idx], self.y[batch_idx]
    
    def __len__(self) -> int:
        """
        Description:
        ---------------
            Возвращает количество батчей в одной эпохе.
        
        Returns:
        ---------------
            Количество батчей.
        """
        return self.n_batches


class Dataset:
    """
    Description:
    ---------------
        Базовый класс для датасетов.
        Определяет интерфейс для всех датасетов.
    
    Args:
    ---------------
        Нет аргументов при инициализации.
    
    Returns:
    ---------------
        Нет возвращаемых значений.
    
    Raises:
    ---------------
        NotImplementedError: Вызывается, если дочерний класс не реализует
                            необходимые методы.
    """
    
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Description:
        ---------------
            Загружает и подготавливает данные для обучения и тестирования.
        
        Returns:
        ---------------
            Кортеж из четырех элементов: (X_train, X_test, y_train, y_test).
        
        Raises:
        ---------------
            NotImplementedError: Должен быть реализован в дочернем классе.
        """
        raise NotImplementedError


class IrisDataset(Dataset):
    """
    Description:
    ---------------
        Датасет Iris (ирисы Фишера).
        Содержит данные о цветках ириса трех видов.
    
    Args:
    ---------------
        Нет аргументов при инициализации.
    
    Returns:
    ---------------
        Нет возвращаемых значений.
    
    Examples:
    ---------------
        >>> dataset = IrisDataset()
        >>> X_train, X_test, y_train, y_test = dataset.load_data()
    """
    
    def load_data(
        self, 
        test_size: float = 0.2, 
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Description:
        ---------------
            Загружает и подготавливает данные Iris.
        
        Args:
        ---------------
            test_size: Доля тестовых данных (от 0 до 1).
            random_state: Seed для генератора случайных чисел.
        
        Returns:
        ---------------
            Кортеж из четырех элементов: (X_train, X_test, y_train, y_test).
        """
        data = load_iris()
        X, y = data.data, data.target
        
        # Нормализация данных
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # One-hot encoding для меток
        encoder = OneHotEncoder(sparse_output=False)
        y = encoder.fit_transform(y.reshape(-1, 1))
        
        # Разделение на тренировочную и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        return X_train, X_test, y_train, y_test


class MNISTDataset(Dataset):
    """
    Description:
    ---------------
        Датасет MNIST.
        Содержит изображения рукописных цифр от 0 до 9.
    
    Args:
    ---------------
        Нет аргументов при инициализации.
    
    Returns:
    ---------------
        Нет возвращаемых значений.
    
    Examples:
    ---------------
        >>> dataset = MNISTDataset()
        >>> X_train, X_test, y_train, y_test = dataset.load_data()
    """
    
    def load_data(
        self, 
        test_size: int = 10000, 
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Description:
        ---------------
            Загружает и подготавливает данные MNIST.
        
        Args:
        ---------------
            test_size: Размер тестовой выборки.
            random_state: Seed для генератора случайных чисел.
        
        Returns:
        ---------------
            Кортеж из четырех элементов: (X_train, X_test, y_train, y_test).
        """
        # Загрузка MNIST с помощью fetch_openml
        mnist = fetch_openml('mnist_784', version=1, parser='auto')
        X, y = mnist.data.astype('float32'), mnist.target.astype('int')
        
        # Нормализация данных и преобразование в формат 0-1
        X = X / 255.0
        
        # One-hot encoding для меток
        encoder = OneHotEncoder(sparse_output=False)
        y = encoder.fit_transform(y.reshape(-1, 1))
        
        # Разделение на тренировочную и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        return X_train, X_test, y_train, y_test


class FashionMNISTDataset(Dataset):
    """
    Description:
    ---------------
        Датасет Fashion MNIST.
        Содержит изображения предметов одежды 10 категорий.
    
    Args:
    ---------------
        Нет аргументов при инициализации.
    
    Returns:
    ---------------
        Нет возвращаемых значений.
    
    Examples:
    ---------------
        >>> dataset = FashionMNISTDataset()
        >>> X_train, X_test, y_train, y_test = dataset.load_data()
    """
    
    def load_data(
        self, 
        test_size: int = 10000, 
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Description:
        ---------------
            Загружает и подготавливает данные Fashion MNIST.
        
        Args:
        ---------------
            test_size: Размер тестовой выборки.
            random_state: Seed для генератора случайных чисел.
        
        Returns:
        ---------------
            Кортеж из четырех элементов: (X_train, X_test, y_train, y_test).
        """
        # Загрузка Fashion MNIST с помощью fetch_openml
        fashion_mnist = fetch_openml('Fashion-MNIST', version=1, parser='auto')
        X, y = fashion_mnist.data.astype('float32'), fashion_mnist.target.astype('int')
        
        # Нормализация данных и преобразование в формат 0-1
        X = X / 255.0
        
        # One-hot encoding для меток
        encoder = OneHotEncoder(sparse_output=False)
        y = encoder.fit_transform(y.reshape(-1, 1))
        
        # Разделение на тренировочную и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        return X_train, X_test, y_train, y_test


class NeuralNetwork:
    """
    Description:
    ---------------
        Основной класс нейронной сети.
        Предоставляет функционал для создания, обучения и использования
        нейронных сетей.
    
    Args:
    ---------------
        Нет аргументов при инициализации.
    
    Returns:
    ---------------
        Нет возвращаемых значений.
    
    Examples:
    ---------------
        >>> nn = NeuralNetwork()
        >>> nn.add(Dense(10, 5, activation=ReLU()))
        >>> nn.add(Dense(5, 2, activation=Softmax()))
        >>> nn.compile(loss=CrossEntropy(), optimizer=Adam())
        >>> nn.fit(X_train, y_train, epochs=10)
    """
    
    def __init__(self) -> None:
        """
        Description:
        ---------------
            Инициализирует нейронную сеть.
        """
        self.layers = []
        self.loss = None
        self.optimizer = None
    
    def add(self, layer: Layer) -> None:
        """
        Description:
        ---------------
            Добавляет слой в нейронную сеть.
        
        Args:
        ---------------
            layer: Слой для добавления в нейронную сеть.
        """
        self.layers.append(layer)
    
    def compile(self, loss: Loss, optimizer: Optimizer) -> None:
        """
        Description:
        ---------------
            Компилирует модель с выбором функции потерь и оптимизатора.
        
        Args:
        ---------------
            loss: Функция потерь для обучения.
            optimizer: Оптимизатор для обучения.
        """
        self.loss = loss
        self.optimizer = optimizer
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Description:
        ---------------
            Выполняет прямой проход по нейронной сети.
        
        Args:
        ---------------
            X: Входные данные.
        
        Returns:
        ---------------
            Выходные данные нейронной сети.
        """
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def backward(self, grad: np.ndarray) -> None:
        """
        Description:
        ---------------
            Выполняет обратный проход по нейронной сети.
        
        Args:
        ---------------
            grad: Градиент от функции потерь.
        """
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
    
    def train_on_batch(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Description:
        ---------------
            Обучение на одном батче данных.
        
        Args:
        ---------------
            X: Входные данные батча.
            y: Целевые значения батча.
        
        Returns:
        ---------------
            Значение функции потерь на батче.
        """
        # Прямой проход
        y_pred = self.forward(X)
        
        # Вычисление функции потерь
        loss_value = self.loss.forward(y_pred, y)
        
        # Обратный проход
        grad = self.loss.backward(y_pred, y)
        self.backward(grad)
        
        # Обновление весов
        for layer in self.layers:
            self.optimizer.update(layer)
        
        return loss_value
    
    def fit(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray, 
        X_val: Optional[np.ndarray] = None, 
        y_val: Optional[np.ndarray] = None, 
        epochs: int = 10, 
        batch_size: int = 32, 
        verbose: int = 1, 
        early_stopping: bool = False, 
        patience: int = 5
    ) -> Dict[str, List[float]]:
        """
        Description:
        ---------------
            Обучение нейронной сети.
        
        Args:
        ---------------
            X_train: Входные данные для обучения.
            y_train: Целевые значения для обучения.
            X_val: Входные данные для валидации (опционально).
            y_val: Целевые значения для валидации (опционально).
            epochs: Количество эпох обучения.
            batch_size: Размер мини-батча.
            verbose: Уровень подробности вывода (0, 1).
            early_stopping: Флаг раннего останова.
            patience: Количество эпох ожидания улучшения.
        
        Returns:
        ---------------
            Словарь с историей обучения (потери на обучающей 
            и валидационной выборках).
        """
        n_samples = X_train.shape[0]
        train_losses = []
        val_losses = []
        
        # Для early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Создаем загрузчик данных для этой эпохи
            data_loader = DataLoader(X_train, y_train, batch_size=batch_size)
            
            # Включаем режим обучения для слоев с разными режимами
            for layer in self.layers:
                if hasattr(layer, 'training'):
                    layer.training = True
            
            # Обучение на батчах
            epoch_losses = []
            
            if verbose:
                iterator = tqdm(data_loader, desc=f"Epoch {epoch+1}/{epochs}")
            else:
                iterator = data_loader
            
            for X_batch, y_batch in iterator:
                loss_value = self.train_on_batch(X_batch, y_batch)
                epoch_losses.append(loss_value)
            
            avg_loss = np.mean(epoch_losses)
            train_losses.append(avg_loss)
            
            # Валидация, если предоставлены валидационные данные
            val_loss = None
            if X_val is not None and y_val is not None:
                # Включаем режим вывода для слоев с разными режимами
                for layer in self.layers:
                    if hasattr(layer, 'training'):
                        layer.training = False
                
                y_val_pred = self.forward(X_val)
                val_loss = self.loss.forward(y_val_pred, y_val)
                val_losses.append(val_loss)
                
                # Early stopping
                if early_stopping:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            print(f"\nEarly stopping at epoch {epoch+1}")
                            break
            
            # Вывод прогресса
            if verbose:
                status = f"loss: {avg_loss:.4f}"
                if val_loss is not None:
                    status += f" - val_loss: {val_loss:.4f}"
                print(status)
        
        return {"train_loss": train_losses, "val_loss": val_losses}
    
    def evaluate(
        self, X: np.ndarray, y: np.ndarray
    ) -> Union[float, Tuple[float, float]]:
        """
        Description:
        ---------------
            Оценка модели на данных.
        
        Args:
        ---------------
            X: Входные данные для оценки.
            y: Целевые значения для оценки.
        
        Returns:
        ---------------
            Значение функции потерь или кортеж (потеря, точность) 
            для задач классификации.
        """
        # Включаем режим вывода для слоев с разными режимами
        for layer in self.layers:
            if hasattr(layer, 'training'):
                layer.training = False
        
        y_pred = self.forward(X)
        loss_value = self.loss.forward(y_pred, y)
        
        # Для классификации также вычисляем точность
        if isinstance(self.loss, (CrossEntropy, BinaryCrossEntropy)):
            if y.shape[1] > 1:  # Многоклассовая классификация
                accuracy = np.mean(
                    np.argmax(y_pred, axis=1) == np.argmax(y, axis=1)
                )
            else:  # Бинарная классификация
                accuracy = np.mean((y_pred > 0.5) == y)
            
            return loss_value, accuracy
        
        return loss_value
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Description:
        ---------------
            Предсказание на новых данных.
        
        Args:
        ---------------
            X: Входные данные для предсказания.
        
        Returns:
        ---------------
            Предсказанные значения.
        """
        # Включаем режим вывода для слоев с разными режимами
        for layer in self.layers:
            if hasattr(layer, 'training'):
                layer.training = False
        
        return self.forward(X)
    
    def save(self, filepath: str) -> None:
        """
        Description:
        ---------------
            Сохранение модели в файл.
        
        Args:
        ---------------
            filepath: Путь для сохранения модели.
        """
        model_data = {
            'layers': self.layers,
            'loss': self.loss,
            'optimizer': self.optimizer
        }
        
        with gzip.open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load(self, filepath: str) -> None:
            """
            Description:
            ---------------
                Загрузка модели из файла.
            
            Args:
            ---------------
                filepath: Путь к сохраненной модели.
            
            Returns:
            ---------------
                Нет возвращаемых значений.
            
            Raises:
            ---------------
                FileNotFoundError: Возникает, если файл не найден.
                pickle.UnpicklingError: Возникает при ошибке десериализации.
            
            Examples:
            ---------------
                >>> nn = NeuralNetwork()
                >>> nn.load('model.gz')
            """
            with gzip.open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.layers = model_data['layers']
            self.loss = model_data['loss']
            self.optimizer = model_data['optimizer']