import numpy as np
import pandas as pd
from typing import List, Tuple, Callable, Union, Dict, Any, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import copy

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


class Model:
    """Класс нейронной сети"""
    def __init__(self):
        self.layers = []
        self.loss_fn = None
        self.optimizer = None
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }

    def add(self, layer: Layer):
        """Добавление слоя в нейронную сеть"""
        self.layers.append(layer)

    def compile(self, loss: Loss, optimizer: Optimizer):
        """Компиляция модели: выбор функции потерь и оптимизатора"""
        self.loss_fn = loss
        self.optimizer = optimizer

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Прямой проход через все слои сети"""
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad: np.ndarray):
        """Обратный проход для вычисления градиентов"""
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def update_params(self):
        """Обновление параметров всех слоев"""
        for layer in self.layers:
            layer.update_params(self.optimizer)

    def train(self):
        """Включение режима обучения для всех слоев"""
        for layer in self.layers:
            if hasattr(layer, 'train'):
                layer.train()

    def eval(self):
        """Включение режима тестирования для всех слоев"""
        for layer in self.layers:
            if hasattr(layer, 'eval'):
                layer.eval()

    def _calculate_accuracy(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Вычисление точности предсказаний"""
        # Для задачи классификации
        if y_pred.shape[1] > 1:  # Многоклассовая классификация
            pred_classes = np.argmax(y_pred, axis=1)
            
            # Проверяем формат y_true
            if len(y_true.shape) > 1 and y_true.shape[1] > 1:  # one-hot формат
                true_classes = np.argmax(y_true, axis=1)
            else:  # индексы классов
                true_classes = y_true
                
            return np.mean(pred_classes == true_classes)
        else:  # Бинарная классификация
            # Округляем предсказания для бинарной классификации
            pred_classes = (y_pred > 0.5).astype(int)
            return np.mean(pred_classes == y_true)

    def fit(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray, 
        epochs: int = 10, 
        batch_size: int = 32, 
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        verbose: bool = True,
        shuffle: bool = True,
        callback: Optional[Callable[[dict], None]] = None
    ):
        """Обучение нейронной сети"""
        num_samples = X_train.shape[0]
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # Перемешиваем данные, если нужно
            if shuffle:
                indices = np.random.permutation(num_samples)
                X_train = X_train[indices]
                y_train = y_train[indices]
            
            # Инициализируем потери и точность для текущей эпохи
            epoch_loss = 0.0
            epoch_acc = 0.0
            
            # Включаем режим обучения
            self.train()
            
            # Обучение по мини-батчам
            num_batches = (num_samples + batch_size - 1) // batch_size  # ceil division
            
            if verbose:
                batch_iterator = tqdm(range(num_batches), desc=f"Эпоха {epoch+1}/{epochs}")
            else:
                batch_iterator = range(num_batches)
            
            for batch_idx in batch_iterator:
                # Получаем текущий батч
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, num_samples)
                X_batch = X_train[start_idx:end_idx]
                y_batch = y_train[start_idx:end_idx]
                
                # Прямой проход
                y_pred = self.forward(X_batch)
                
                # Вычисляем потери
                batch_loss = self.loss_fn(y_pred, y_batch)
                
                # Вычисляем градиент функции потерь
                loss_grad = self.loss_fn.derivative(y_pred, y_batch)
                
                # Обратный проход
                self.backward(loss_grad)
                
                # Обновляем параметры
                self.update_params()
                
                # Накапливаем потери и точность
                epoch_loss += batch_loss * (end_idx - start_idx) / num_samples
                epoch_acc += self._calculate_accuracy(y_pred, y_batch) * (end_idx - start_idx) / num_samples
            
            # Сохраняем потери и точность для текущей эпохи
            self.history['train_loss'].append(epoch_loss)
            self.history['train_acc'].append(epoch_acc)
            
            # Проверяем на валидационных данных, если они предоставлены
            val_loss = None
            val_acc = None
            
            if validation_data is not None:
                X_val, y_val = validation_data
                
                # Включаем режим тестирования
                self.eval()
                
                # Прямой проход на валидационных данных
                y_val_pred = self.forward(X_val)
                
                # Вычисляем потери и точность на валидационных данных
                val_loss = self.loss_fn(y_val_pred, y_val)
                val_acc = self._calculate_accuracy(y_val_pred, y_val)
                
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
            
            # Вычисляем время выполнения эпохи
            epoch_time = time.time() - epoch_start_time
            
            # Выводим информацию о текущей эпохе
            if verbose:
                status = f"Эпоха {epoch+1}/{epochs} - {epoch_time:.2f}s - loss: {epoch_loss:.4f} - acc: {epoch_acc:.4f}"
                if val_loss is not None:
                    status += f" - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}"
                print(status)
            
            # Вызываем callback, если предоставлен
            if callback is not None:
                callback_info = {
                    'epoch': epoch + 1,
                    'train_loss': epoch_loss,
                    'train_acc': epoch_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc
                }
                callback(callback_info)
        
        return self.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Предсказание на новых данных"""
        # Включаем режим тестирования
        self.eval()
        
        # Прямой проход
        return self.forward(X)

    def save_weights(self, filename: str):
        """Сохранение весов модели в файл"""
        weights = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'params') and layer.params:
                weights[f'layer_{i}'] = {key: value for key, value in layer.params.items()}
        
        np.savez(filename, **weights)

    def load_weights(self, filename: str):
        """Загрузка весов модели из файла"""
        weights = np.load(filename, allow_pickle=True)
        
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'params') and layer.params:
                layer_key = f'layer_{i}'
                if layer_key in weights:
                    for param_name in layer.params:
                        if param_name in weights[layer_key].item():
                            layer.params[param_name] = weights[layer_key].item()[param_name]

    def summary(self):
        """Вывод сводки о модели"""
        print("Структура модели:")
        print("-" * 80)
        print(f"{'Слой (тип)':<30} {'Выходная форма':<20} {'Параметры':<20}")
        print("=" * 80)
        
        total_params = 0
        
        for i, layer in enumerate(self.layers):
            layer_name = f"{i}: {layer.__class__.__name__}"
            
            # Определяем выходную форму и количество параметров
            output_shape = "?"
            params = 0
            
            if isinstance(layer, Dense):
                output_shape = f"({layer.output_size},)"
                params = layer.input_size * layer.output_size
                if layer.use_bias:
                    params += layer.output_size
            elif isinstance(layer, BatchNormalization):
                output_shape = f"({layer.input_size},)"
                params = 2 * layer.input_size  # gamma и beta
            
            print(f"{layer_name:<30} {output_shape:<20} {params:<20}")
            total_params += params
        
        print("-" * 80)
        print(f"Всего параметров: {total_params}")
        print("-" * 80)

    def plot_history(self):
        """Построение графиков обучения"""
        plt.figure(figsize=(12, 4))
        
        # График потерь
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='Потери на обучении')
        if 'val_loss' in self.history and self.history['val_loss']:
            plt.plot(self.history['val_loss'], label='Потери на валидации')
        plt.title('График потерь')
        plt.xlabel('Эпоха')
        plt.ylabel('Потери')
        plt.legend()
        
        # График точности
        plt.subplot(1, 2, 2)
        plt.plot(self.history['train_acc'], label='Точность на обучении')
        if 'val_acc' in self.history and self.history['val_acc']:
            plt.plot(self.history['val_acc'], label='Точность на валидации')
        plt.title('График точности')
        plt.xlabel('Эпоха')
        plt.ylabel('Точность')
        plt.legend()
        
        plt.tight_layout()
        plt.show()


# Утилиты для работы с данными
class DataUtils:
    @staticmethod
    def normalize(X: np.ndarray) -> np.ndarray:
        """Нормализация данных (значения от 0 до 1)"""
        return X / 255.0 if X.max() > 1 else X

    @staticmethod
    def standardize(X: np.ndarray) -> np.ndarray:
        """Стандартизация данных (среднее = 0, стандартное отклонение = 1)"""
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        return (X - mean) / (std + 1e-7)  # Добавляем эпсилон для избежания деления на 0

    @staticmethod
    def one_hot_encode(y: np.ndarray, num_classes: int) -> np.ndarray:
        """Преобразование меток классов в one-hot формат"""
        return np.eye(num_classes)[y.astype(int)]

    @staticmethod
    def train_test_split(
        X: np.ndarray, 
        y: np.ndarray, 
        test_size: float = 0.2, 
        random_state: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Разделение данных на обучающую и тестовую выборки"""
        if random_state is not None:
            np.random.seed(random_state)
        
        num_samples = X.shape[0]
        indices = np.random.permutation(num_samples)
        test_count = int(num_samples * test_size)
        
        test_indices = indices[:test_count]
        train_indices = indices[test_count:]
        
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        
        return X_train, X_test, y_train, y_test

    @staticmethod
    def batch_generator(
        X: np.ndarray, 
        y: np.ndarray, 
        batch_size: int, 
        shuffle: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Генератор батчей для обучения"""
        num_samples = X.shape[0]
        indices = np.arange(num_samples)
        
        if shuffle:
            np.random.shuffle(indices)
        
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]
            
            yield X[batch_indices], y[batch_indices]

    @staticmethod
    def k_fold_split(
        X: np.ndarray, 
        y: np.ndarray, 
        n_folds: int, 
        random_state: Optional[int] = None
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Разделение данных на k разбиений для кросс-валидации"""
        if random_state is not None:
            np.random.seed(random_state)
        
        num_samples = X.shape[0]
        indices = np.random.permutation(num_samples)
        
        fold_size = num_samples // n_folds
        folds = []
        
        for i in range(n_folds):
            start_idx = i * fold_size
            end_idx = (i + 1) * fold_size if i < n_folds - 1 else num_samples
            
            val_indices = indices[start_idx:end_idx]
            train_indices = np.concatenate([indices[:start_idx], indices[end_idx:]])
            
            X_train, X_val = X[train_indices], X[val_indices]
            y_train, y_val = y[train_indices], y[val_indices]
            
            folds.append((X_train, X_val, y_train, y_val))
        
        return folds

    @staticmethod
    def data_augmentation(
        X: np.ndarray, 
        y: np.ndarray, 
        augmentation_functions: List[Callable[[np.ndarray], np.ndarray]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Аугментация данных с помощью заданных функций преобразования"""
        X_augmented = [X]
        y_augmented = [y]
        
        for aug_func in augmentation_functions:
            X_aug = np.array([aug_func(x) for x in X])
            X_augmented.append(X_aug)
            y_augmented.append(y)
        
        return np.vstack(X_augmented), np.concatenate(y_augmented)

    @staticmethod
    def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
        """Поворот изображения на заданный угол"""
        from scipy.ndimage import rotate
        return rotate(image, angle, reshape=False)

    @staticmethod
    def flip_image_horizontal(image: np.ndarray) -> np.ndarray:
        """Отражение изображения по горизонтали"""
        return np.fliplr(image)

    @staticmethod
    def add_noise(image: np.ndarray, noise_level: float = 0.1) -> np.ndarray:
        """Добавление шума к изображению"""
        noise = np.random.normal(0, noise_level, image.shape)
        noisy_image = image + noise
        return np.clip(noisy_image, 0, 1) if image.max() <= 1 else np.clip(noisy_image, 0, 255)

# Пример использования: реализация многослойного перцептрона для MNIST
def create_mnist_model():
    """Создание модели для распознавания рукописных цифр (MNIST)"""
    model = Model()
    
    # Входной слой: 784 нейрона (28x28 пикселей)
    # Скрытый слой 1: 128 нейронов с ReLU активацией
    model.add(Dense(784, 128, activation=ReLU(), weight_init='he'))
    model.add(BatchNormalization(128))
    model.add(Dropout(0.3))
    
    # Скрытый слой 2: 64 нейрона с ReLU активацией
    model.add(Dense(128, 64, activation=ReLU(), weight_init='he'))
    model.add(BatchNormalization(64))
    model.add(Dropout(0.3))
    
    # Выходной слой: 10 нейронов (по одному на цифру) с Softmax активацией
    model.add(Dense(64, 10, activation=Softmax(), weight_init='he'))
    
    # Компиляция модели
    model.compile(
        loss=CategoricalCrossEntropy(),
        optimizer=Adam(learning_rate=0.001)
    )
    
    return model

# Пример использования: реализация многослойного перцептрона для Iris
def create_iris_model():
    """Создание модели для классификации ирисов (Iris dataset)"""
    model = Model()
    
    # Входной слой: 4 нейрона (количество признаков в датасете Iris)
    # Скрытый слой: 10 нейронов с ReLU активацией
    model.add(Dense(4, 10, activation=ReLU(), weight_init='he'))
    model.add(BatchNormalization(10))
    
    # Выходной слой: 3 нейрона (по одному на класс ириса) с Softmax активацией
    model.add(Dense(10, 3, activation=Softmax(), weight_init='he'))
    
    # Компиляция модели
    model.compile(
        loss=CategoricalCrossEntropy(),
        optimizer=MomentumSGD(learning_rate=0.01, momentum=0.9)
    )
    
    return model

# Пример использования: реализация регрессионной модели
def create_regression_model(input_size: int):
    """Создание модели для задачи регрессии"""
    model = Model()
    
    # Входной слой: input_size нейронов
    # Скрытый слой 1: 32 нейрона с ReLU активацией
    model.add(Dense(input_size, 32, activation=ReLU(), weight_init='he'))
    model.add(BatchNormalization(32))
    model.add(Dropout(0.2))
    
    # Скрытый слой 2: 16 нейронов с ReLU активацией
    model.add(Dense(32, 16, activation=ReLU(), weight_init='he'))
    model.add(BatchNormalization(16))
    
    # Выходной слой: 1 нейрон без активации (для регрессии)
    model.add(Dense(16, 1, activation=None))
    
    # Компиляция модели
    # Используем Gradient Clipping для предотвращения взрывающихся градиентов
    base_optimizer = RMSprop(learning_rate=0.001, decay_rate=0.9)
    optimizer_with_clipping = GradientClipping(base_optimizer, clip_value=1.0)
    
    model.compile(
        loss=MSE(),
        optimizer=optimizer_with_clipping
    )
    
    return model

# Пример использования на датасете MNIST
def mnist_example():
    """Пример использования фреймворка для классификации рукописных цифр MNIST"""
    # Загрузка данных MNIST
    from sklearn.datasets import fetch_openml
    
    print("Загрузка датасета MNIST...")
    mnist = fetch_openml('mnist_784', version=1, parser='auto')
    X = mnist.data.astype(np.float32).values
    y = mnist.target.astype(np.int32).values
    
    # Предобработка данных
    X = DataUtils.normalize(X)  # Нормализация пикселей (от 0 до 1)
    y_one_hot = DataUtils.one_hot_encode(y, 10)  # Преобразование в one-hot формат
    
    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = DataUtils.train_test_split(
        X, y_one_hot, test_size=0.2, random_state=42
    )
    
    print(f"Размер обучающей выборки: {X_train.shape}")
    print(f"Размер тестовой выборки: {X_test.shape}")
    
    # Создание модели
    model = create_mnist_model()
    model.summary()
    
    # Обучение модели
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=128,
        validation_data=(X_test, y_test),
        verbose=True,
        shuffle=True
    )
    
    # Оценка модели
    y_pred = model.predict(X_test)
    accuracy = model._calculate_accuracy(y_pred, y_test)
    print(f"Точность на тестовой выборке: {accuracy:.4f}")
    
    # Построение графиков обучения
    model.plot_history()
    
    return model, history

# Пример использования на датасете Iris
def iris_example():
    """Пример использования фреймворка для классификации цветков Iris"""
    # Загрузка данных Iris
    from sklearn.datasets import load_iris
    
    print("Загрузка датасета Iris...")
    iris = load_iris()
    X = iris.data.astype(np.float32)
    y = iris.target
    
    # Предобработка данных
    X = DataUtils.standardize(X)  # Стандартизация признаков
    y_one_hot = DataUtils.one_hot_encode(y, 3)  # Преобразование в one-hot формат
    
    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = DataUtils.train_test_split(
        X, y_one_hot, test_size=0.2, random_state=42
    )
    
    print(f"Размер обучающей выборки: {X_train.shape}")
    print(f"Размер тестовой выборки: {X_test.shape}")
    
    # Создание модели
    model = create_iris_model()
    model.summary()
    
    # Обучение модели
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=16,
        validation_data=(X_test, y_test),
        verbose=True,
        shuffle=True
    )
    
    # Оценка модели
    y_pred = model.predict(X_test)
    accuracy = model._calculate_accuracy(y_pred, y_test)
    print(f"Точность на тестовой выборке: {accuracy:.4f}")
    
    # Построение графиков обучения
    model.plot_history()
    
    return model, history

# Пример использования для задачи регрессии
def regression_example():
    """Пример использования фреймворка для задачи регрессии"""
    # Создание синтетических данных
    np.random.seed(42)
    X = np.random.rand(1000, 5)  # 1000 примеров, 5 признаков
    
    # Создаем целевую переменную с нелинейной зависимостью и шумом
    y = 3 * X[:, 0]**2 + 2 * X[:, 1] - 1.5 * X[:, 2] + 0.5 * X[:, 3]**3 - X[:, 4] + np.random.normal(0, 0.1, 1000)
    y = y.reshape(-1, 1)  # Форма (1000, 1)
    
    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = DataUtils.train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Размер обучающей выборки: {X_train.shape}")
    print(f"Размер тестовой выборки: {X_test.shape}")
    
    # Создание модели
    model = create_regression_model(input_size=5)
    model.summary()
    
    # Обучение модели
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=True,
        shuffle=True
    )
    
    # Оценка модели
    y_pred = model.predict(X_test)
    mse = np.mean(np.square(y_pred - y_test))
    print(f"Среднеквадратичная ошибка на тестовой выборке: {mse:.4f}")
    
    # Построение графиков обучения
    model.plot_history()
    
    # Визуализация результатов регрессии
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Истинные значения')
    plt.ylabel('Предсказанные значения')
    plt.title('Сравнение истинных и предсказанных значений')
    plt.show()
    
    return model, history

# Демонстрация обучения "в несколько строк"
def quick_start_example():
    """Демонстрация быстрого создания и обучения модели"""
    # Создание синтетических данных для бинарной классификации
    np.random.seed(42)
    X = np.random.randn(500, 2)  # 500 примеров, 2 признака
    y = (X[:, 0]**2 + X[:, 1]**2 < 1).astype(int).reshape(-1, 1)  # Круг
    
    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = DataUtils.train_test_split(X, y, test_size=0.2)
    
    # Создание и обучение модели "в несколько строк"
    model = Model()
    model.add(Dense(2, 16, activation=ReLU()))
    model.add(Dense(16, 8, activation=ReLU()))
    model.add(Dense(8, 1, activation=Sigmoid()))
    model.compile(loss=BinaryCrossEntropy(), optimizer=Adam())
    
    # Обучение модели
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))
    
    # Визуализация результатов
    plt.figure(figsize=(10, 8))
    
    # Создаем сетку для визуализации разделяющей границы
    h = 0.01
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Получаем предсказания для всех точек сетки
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = (Z > 0.5).astype(int).reshape(xx.shape)
    
    # Визуализируем разделяющую границу
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
    
    # Визуализируем обучающие данные
    plt.scatter(X_train[y_train.ravel() == 0, 0], X_train[y_train.ravel() == 0, 1], c='blue', label='Класс 0 (трейн)')
    plt.scatter(X_train[y_train.ravel() == 1, 0], X_train[y_train.ravel() == 1, 1], c='red', label='Класс 1 (трейн)')
    
    # Визуализируем тестовые данные
    plt.scatter(X_test[y_test.ravel() == 0, 0], X_test[y_test.ravel() == 0, 1], c='blue', marker='x', label='Класс 0 (тест)')
    plt.scatter(X_test[y_test.ravel() == 1, 0], X_test[y_test.ravel() == 1, 1], c='red', marker='x', label='Класс 1 (тест)')
    
    plt.title('Классификация с помощью нейронной сети')
    plt.xlabel('Признак 1')
    plt.ylabel('Признак 2')
    plt.legend()
    plt.show()
    
    return model

# Пример использования кросс-валидации
def cross_validation_example():
    """Пример использования кросс-валидации"""
    # Загрузка данных Iris
    from sklearn.datasets import load_iris
    
    print("Загрузка датасета Iris для кросс-валидации...")
    iris = load_iris()
    X = iris.data.astype(np.float32)
    y = iris.target
    
    # Предобработка данных
    X = DataUtils.standardize(X)  # Стандартизация признаков
    y_one_hot = DataUtils.one_hot_encode(y, 3)  # Преобразование в one-hot формат
    
    # Кросс-валидация (5 разбиений)
    n_folds = 5
    folds = DataUtils.k_fold_split(X, y_one_hot, n_folds=n_folds, random_state=42)
    
    val_accuracies = []
    
    for fold_idx, (X_train, X_val, y_train, y_val) in enumerate(folds):
        print(f"\nОбучение на разбиении {fold_idx+1}/{n_folds}")
        
        # Создание модели
        model = create_iris_model()
        
        # Обучение модели
        model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=16,
            validation_data=(X_val, y_val),
            verbose=False  # Отключаем вывод прогресса для краткости
        )
        
        # Оценка модели на валидационной выборке
        y_pred = model.predict(X_val)
        accuracy = model._calculate_accuracy(y_pred, y_val)
        val_accuracies.append(accuracy)
        
        print(f"Точность на валидационной выборке (разбиение {fold_idx+1}): {accuracy:.4f}")
    
    # Вычисляем среднюю точность по всем разбиениям
    mean_accuracy = np.mean(val_accuracies)
    std_accuracy = np.std(val_accuracies)
    
    print(f"\nСредняя точность по всем разбиениям: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    
    return val_accuracies

# Демонстрация использования колбэков
def early_stopping_callback(info: dict, patience: int = 5, min_delta: float = 0.001):
    """Колбэк для ранней остановки обучения"""
    if not hasattr(early_stopping_callback, 'best_val_loss'):
        early_stopping_callback.best_val_loss = float('inf')
        early_stopping_callback.wait = 0
        early_stopping_callback.stopped_epoch = 0
        early_stopping_callback.stop_training = False
    
    current_val_loss = info['val_loss']
    
    if early_stopping_callback.stop_training:
        return True
    
    if current_val_loss < early_stopping_callback.best_val_loss - min_delta:
        early_stopping_callback.best_val_loss = current_val_loss
        early_stopping_callback.wait = 0
    else:
        early_stopping_callback.wait += 1
        if early_stopping_callback.wait >= patience:
            early_stopping_callback.stopped_epoch = info['epoch']
            early_stopping_callback.stop_training = True
            print(f"\nРанняя остановка на эпохе {info['epoch']}")
            return True
    
    return False

def early_stopping_example():
    """Пример использования ранней остановки обучения"""
    # Создание синтетических данных для регрессии
    np.random.seed(42)
    X = np.random.rand(500, 3)  # 500 примеров, 3 признака
    y = 2 * X[:, 0] - 1 * X[:, 1] + 0.5 * X[:, 2] + np.random.normal(0, 0.1, 500)
    y = y.reshape(-1, 1)
    
    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = DataUtils.train_test_split(X, y, test_size=0.2)
    
    # Создание модели
    model = Model()
    model.add(Dense(3, 32, activation=ReLU()))
    model.add(Dense(32, 16, activation=ReLU()))
    model.add(Dense(16, 1, activation=None))
    model.compile(loss=MSE(), optimizer=Adam(learning_rate=0.01))
    
    # Сброс состояния колбэка
    if hasattr(early_stopping_callback, 'stop_training'):
        delattr(early_stopping_callback, 'best_val_loss')
        delattr(early_stopping_callback, 'wait')
        delattr(early_stopping_callback, 'stopped_epoch')
        delattr(early_stopping_callback, 'stop_training')
    
    # Создаем обертку для колбэка с параметрами
    def callback_wrapper(info):
        should_stop = early_stopping_callback(info, patience=3, min_delta=0.0001)
        if should_stop:
            model.train_interrupted = True
    
    # Добавляем флаг для отслеживания прерывания обучения
    model.train_interrupted = False
    
    # Обучение модели с ранней остановкой
    history = model.fit(
        X_train, y_train,
        epochs=100,  # Максимальное количество эпох
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=True,
        callback=callback_wrapper
    )
    
    # Проверяем, было ли прервано обучение
    if model.train_interrupted:
        print(f"Обучение было остановлено на эпохе {early_stopping_callback.stopped_epoch}")
    
    # Оценка модели
    y_pred = model.predict(X_test)
    mse = np.mean(np.square(y_pred - y_test))
    print(f"Среднеквадратичная ошибка на тестовой выборке: {mse:.4f}")
    
    # Построение графиков обучения
    model.plot_history()
    
    return model, history

# Пример обучения с использованием различных оптимизаторов
def optimizer_comparison():
    """Сравнение различных оптимизаторов на одной задаче"""
    # Загрузка данных Iris
    from sklearn.datasets import load_iris
    
    print("Загрузка датасета Iris для сравнения оптимизаторов...")
    iris = load_iris()
    X = iris.data.astype(np.float32)
    y = iris.target
    
    # Предобработка данных
    X = DataUtils.standardize(X)  # Стандартизация признаков
    y_one_hot = DataUtils.one_hot_encode(y, 3)  # Преобразование в one-hot формат
    
    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = DataUtils.train_test_split(
        X, y_one_hot, test_size=0.2, random_state=42
    )
    
    # Определяем оптимизаторы для сравнения
    optimizers = {
        "SGD": SGD(learning_rate=0.01),
        "Momentum SGD": MomentumSGD(learning_rate=0.01, momentum=0.9),
        "RMSprop": RMSprop(learning_rate=0.01, decay_rate=0.9),
        "Adam": Adam(learning_rate=0.01),
        "SGD с градиентным клиппингом": GradientClipping(SGD(learning_rate=0.01), clip_value=1.0)
    }
    
    # Обучаем модель с каждым оптимизатором
    histories = {}
    accuracies = {}
    
    for name, optimizer in optimizers.items():
        print(f"\nОбучение с оптимизатором: {name}")
        
        # Создание модели
        model = Model()
        model.add(Dense(4, 10, activation=ReLU(), weight_init='he'))
        model.add(Dense(10, 3, activation=Softmax(), weight_init='he'))
        model.compile(loss=CategoricalCrossEntropy(), optimizer=optimizer)
        
        # Обучение модели
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=16,
            validation_data=(X_test, y_test),
            verbose=False  # Отключаем вывод прогресса для краткости
        )
        
        # Оценка модели
        y_pred = model.predict(X_test)
        accuracy = model._calculate_accuracy(y_pred, y_test)
        
        print(f"Точность на тестовой выборке: {accuracy:.4f}")
        
        histories[name] = history
        accuracies[name] = accuracy
    
    # Визуализация результатов
    plt.figure(figsize=(12, 6))
    
    # График потерь на обучении
    plt.subplot(1, 2, 1)
    for name, history in histories.items():
        plt.plot(history['train_loss'], label=name)
    plt.title('Потери на обучающей выборке')
    plt.xlabel('Эпоха')
    plt.ylabel('Потери')
    plt.legend()
    
    # График точности на валидации
    plt.subplot(1, 2, 2)
    for name, history in histories.items():
        plt.plot(history['val_acc'], label=f"{name} ({accuracies[name]:.4f})")
    plt.title('Точность на валидационной выборке')
    plt.xlabel('Эпоха')
    plt.ylabel('Точность')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return histories, accuracies