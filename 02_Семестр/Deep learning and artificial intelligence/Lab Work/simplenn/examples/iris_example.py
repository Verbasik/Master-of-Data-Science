# simplenn/examples/iris_example.py
"""
Пример использования фреймворка для классификации цветков Iris.
"""

import numpy as np
import matplotlib.pyplot as plt

from core.model import Model
from core.layer import Dense, BatchNormalization
from core.activation import ReLU, Softmax
from core.loss import CategoricalCrossEntropy
from core.optimizer import MomentumSGD
from utils.data_utils import DataUtils

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

def run_iris_example():
    """Запуск примера классификации Iris"""
    try:
        # Загрузка данных Iris
        from sklearn.datasets import load_iris
        
        print("Загрузка датасета Iris...")
        iris = load_iris()
        X = iris.data.astype(np.float32)
        y = iris.target
        feature_names = iris.feature_names
        target_names = iris.target_names
        
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
        print("Начало обучения модели...")
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
        
        # Визуализация границ принятия решений (для первых двух признаков)
        plt.figure(figsize=(10, 8))
        
        # Создаем сетку для визуализации
        h = 0.02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        
        # Создаем входные данные для сетки (используем средние значения для неотображаемых признаков)
        mean_features = np.mean(X, axis=0)
        mesh_points = np.c_[xx.ravel(), yy.ravel(), 
                          np.ones(xx.ravel().shape) * mean_features[2],
                          np.ones(xx.ravel().shape) * mean_features[3]]
        
        # Получаем предсказания для всех точек сетки
        Z = model.predict(mesh_points)
        Z = np.argmax(Z, axis=1).reshape(xx.shape)
        
        # Визуализируем границы принятия решений
        plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Set3)
        
        # Визуализируем обучающие данные
        for i, color in enumerate(['blue', 'red', 'green']):
            idx = np.argmax(y_train, axis=1) == i
            plt.scatter(X_train[idx, 0], X_train[idx, 1], c=color, 
                        label=f'Класс {target_names[i]} (трейн)', edgecolors='k', alpha=0.7)
        
        # Визуализируем тестовые данные
        for i, color in enumerate(['blue', 'red', 'green']):
            idx = np.argmax(y_test, axis=1) == i
            plt.scatter(X_test[idx, 0], X_test[idx, 1], c=color, marker='x', 
                        label=f'Класс {target_names[i]} (тест)', alpha=0.7, s=100)
        
        plt.title('Границы принятия решений для Iris (первые два признака)')
        plt.xlabel(feature_names[0])
        plt.ylabel(feature_names[1])
        plt.legend()
        plt.show()
        
        return model, history
    
    except Exception as e:
        print(f"Ошибка при выполнении примера Iris: {str(e)}")
        return None, None

if __name__ == "__main__":
    print("Запуск примера классификации цветков Iris")
    model, history = run_iris_example()