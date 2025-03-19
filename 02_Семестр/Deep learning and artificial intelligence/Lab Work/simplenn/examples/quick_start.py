# simplenn/examples/quick_start.py
"""
Демонстрация быстрого создания и обучения модели с минимальным кодом.
"""

import numpy as np
import matplotlib.pyplot as plt

from core.model import Model
from core.layer import Dense
from core.activation import Sigmoid, ReLU
from core.loss import BinaryCrossEntropy
from core.optimizer import Adam
from utils.data_utils import DataUtils

def run_quick_start_example():
    """Демонстрация быстрого создания и обучения модели"""
    try:
        print("Создание синтетических данных для бинарной классификации...")
        # Создание синтетических данных для бинарной классификации
        np.random.seed(42)
        X = np.random.randn(500, 2)  # 500 примеров, 2 признака
        y = (X[:, 0]**2 + X[:, 1]**2 < 1).astype(int).reshape(-1, 1)  # Круг
        
        # Разделение на обучающую и тестовую выборки
        X_train, X_test, y_train, y_test = DataUtils.train_test_split(X, y, test_size=0.2)
        
        print(f"Размер обучающей выборки: {X_train.shape}")
        print(f"Размер тестовой выборки: {X_test.shape}")
        
        # Создание и обучение модели "в несколько строк"
        print("Создание и обучение модели...")
        model = Model()
        model.add(Dense(2, 16, activation=ReLU()))
        model.add(Dense(16, 8, activation=ReLU()))
        model.add(Dense(8, 1, activation=Sigmoid()))
        model.compile(loss=BinaryCrossEntropy(), optimizer=Adam())
        
        # Обучение модели
        history = model.fit(
            X_train, y_train, 
            epochs=20, 
            batch_size=32, 
            validation_data=(X_test, y_test),
            verbose=True
        )
        
        # Оценка модели
        y_pred = model.predict(X_test)
        accuracy = model._calculate_accuracy(y_pred, y_test)
        print(f"Точность на тестовой выборке: {accuracy:.4f}")
        
        # Построение графиков обучения
        model.plot_history()
        
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
        plt.scatter(
            X_train[y_train.ravel() == 0, 0], 
            X_train[y_train.ravel() == 0, 1], 
            c='blue', label='Класс 0 (трейн)'
        )
        plt.scatter(
            X_train[y_train.ravel() == 1, 0], 
            X_train[y_train.ravel() == 1, 1], 
            c='red', label='Класс 1 (трейн)'
        )
        
        # Визуализируем тестовые данные
        plt.scatter(
            X_test[y_test.ravel() == 0, 0], 
            X_test[y_test.ravel() == 0, 1], 
            c='blue', marker='x', label='Класс 0 (тест)'
        )
        plt.scatter(
            X_test[y_test.ravel() == 1, 0], 
            X_test[y_test.ravel() == 1, 1], 
            c='red', marker='x', label='Класс 1 (тест)'
        )
        
        plt.title('Классификация с помощью нейронной сети')
        plt.xlabel('Признак 1')
        plt.ylabel('Признак 2')
        plt.legend()
        plt.show()
        
        return model, history
    
    except Exception as e:
        print(f"Ошибка при выполнении быстрого примера: {str(e)}")
        return None, None

if __name__ == "__main__":
    print("Запуск демонстрации быстрого создания и обучения модели")
    model, history = run_quick_start_example()