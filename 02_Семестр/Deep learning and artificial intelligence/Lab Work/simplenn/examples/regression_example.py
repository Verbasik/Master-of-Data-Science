# simplenn/examples/regression_example.py
"""
Пример использования фреймворка для задачи регрессии.
"""

import numpy as np
import matplotlib.pyplot as plt

from core.model import Model
from core.layer import Dense, Dropout, BatchNormalization
from core.activation import ReLU
from core.loss import MSE
from core.optimizer import RMSprop, GradientClipping
from utils.data_utils import DataUtils

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

def run_regression_example():
    """Запуск примера регрессии"""
    try:
        print("Создание синтетических данных для регрессии...")
        # Создание синтетических данных
        np.random.seed(42)
        X = np.random.rand(1000, 5)  # 1000 примеров, 5 признаков
        
        # Создаем целевую переменную с нелинейной зависимостью и шумом
        print("Генерация нелинейной зависимости с шумом...")
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
        print("Начало обучения модели...")
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
        
        # Визуализация важности признаков (приближенно через веса последнего слоя)
        weights = model.layers[-1].params['W']
        feature_importance = np.abs(weights)
        
        plt.figure(figsize=(10, 6))
        plt.bar(np.arange(5), feature_importance.flatten())
        plt.xticks(np.arange(5), [f'Признак {i+1}' for i in range(5)])
        plt.xlabel('Признаки')
        plt.ylabel('Важность (абсолютное значение весов)')
        plt.title('Приближенная важность признаков')
        plt.show()
        
        return model, history
    
    except Exception as e:
        print(f"Ошибка при выполнении примера регрессии: {str(e)}")
        return None, None

if __name__ == "__main__":
    print("Запуск примера регрессии")
    model, history = run_regression_example()