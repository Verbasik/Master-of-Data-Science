# simplenn/examples/mnist_example.py
"""
Пример использования фреймворка для классификации рукописных цифр MNIST.
"""

import numpy as np
import matplotlib.pyplot as plt

from core.model import Model
from core.layer import Dense, Dropout, BatchNormalization
from core.activation import ReLU, Softmax
from core.loss import CategoricalCrossEntropy
from core.optimizer import Adam
from utils.data_utils import DataUtils

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

def run_mnist_example():
    """Запуск примера классификации MNIST"""
    try:
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
        print("Начало обучения модели...")
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
        
        # Визуализация некоторых предсказаний
        plt.figure(figsize=(12, 6))
        for i in range(10):
            plt.subplot(2, 5, i+1)
            plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
            pred_label = np.argmax(y_pred[i])
            true_label = np.argmax(y_test[i])
            plt.title(f"Прогноз: {pred_label}\nИстина: {true_label}")
            plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        return model, history
    
    except Exception as e:
        print(f"Ошибка при выполнении примера MNIST: {str(e)}")
        return None, None

if __name__ == "__main__":
    print("Запуск примера классификации рукописных цифр MNIST")
    model, history = run_mnist_example()