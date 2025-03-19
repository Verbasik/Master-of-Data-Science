# simplenn/core/model.py
"""
Модуль, содержащий основной класс модели нейронной сети.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from typing import List, Tuple, Callable, Dict, Optional

from tqdm import tqdm

from .layer import Layer
from .loss import Loss
from .optimizer import Optimizer

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
            
            if hasattr(layer, 'output_size'):
                output_shape = f"({layer.output_size},)"
                if hasattr(layer, 'input_size'):
                    params = layer.input_size * layer.output_size
                    if hasattr(layer, 'use_bias') and layer.use_bias:
                        params += layer.output_size
            elif hasattr(layer, 'input_size'):
                output_shape = f"({layer.input_size},)"
                if isinstance(layer, BatchNormalization):
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