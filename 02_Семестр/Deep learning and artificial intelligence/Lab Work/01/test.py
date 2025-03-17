import numpy as np
import matplotlib.pyplot as plt
from simplenn import *  # Импортируем наш фреймворк

# ПРИМЕР 1: Классификация MNIST
def run_mnist_example():
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
    
    # Создание модели в несколько строк
    model = Model()
    model.add(Dense(784, 128, activation=ReLU(), weight_init='he'))
    model.add(BatchNormalization(128))
    model.add(Dropout(0.3))
    model.add(Dense(128, 64, activation=ReLU(), weight_init='he'))
    model.add(Dense(64, 10, activation=Softmax(), weight_init='he'))
    model.compile(loss=CategoricalCrossEntropy(), optimizer=Adam(learning_rate=0.001))
    
    # Выводим сводку о модели
    model.summary()
    
    # Обучение модели
    history = model.fit(
        X_train, y_train,
        epochs=10,  # Для демонстрации используем небольшое количество эпох
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

# ПРИМЕР 2: Классификация Iris
def run_iris_example():
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
    
    # Создание модели в несколько строк
    model = Model()
    model.add(Dense(4, 10, activation=ReLU(), weight_init='he'))
    model.add(Dense(10, 3, activation=Softmax(), weight_init='he'))
    model.compile(loss=CategoricalCrossEntropy(), optimizer=MomentumSGD(learning_rate=0.01, momentum=0.9))
    
    # Выводим сводку о модели
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
        plt.scatter(X_train[idx, 0], X_train[idx, 1], c=color, label=f'Класс {i} (трейн)', 
                    edgecolors='k', alpha=0.7)
    
    # Визуализируем тестовые данные
    for i, color in enumerate(['blue', 'red', 'green']):
        idx = np.argmax(y_test, axis=1) == i
        plt.scatter(X_test[idx, 0], X_test[idx, 1], c=color, marker='x', label=f'Класс {i} (тест)', 
                    alpha=0.7, s=100)
    
    plt.title('Границы принятия решений для Iris (первые два признака)')
    plt.xlabel('Длина чашелистика')
    plt.ylabel('Ширина чашелистика')
    plt.legend()
    plt.show()
    
    return model, history

# ПРИМЕР 3: Сравнение оптимизаторов на задаче регрессии
def run_optimizer_comparison():
    # Создание синтетических данных для регрессии
    np.random.seed(42)
    X = np.random.rand(500, 3)  # 500 примеров, 3 признака
    y = 2 * X[:, 0] - 1 * X[:, 1] + 0.5 * X[:, 2] + np.random.normal(0, 0.1, 500)
    y = y.reshape(-1, 1)
    
    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = DataUtils.train_test_split(X, y, test_size=0.2)
    
    # Определяем оптимизаторы для сравнения
    optimizers = {
        "SGD": SGD(learning_rate=0.05),
        "Momentum SGD": MomentumSGD(learning_rate=0.05, momentum=0.9),
        "RMSprop": RMSprop(learning_rate=0.01, decay_rate=0.9),
        "Adam": Adam(learning_rate=0.01),
        "SGD с клиппингом градиентов": GradientClipping(SGD(learning_rate=0.05), clip_value=1.0)
    }
    
    # Обучаем модель с каждым оптимизатором
    histories = {}
    mses = {}
    
    for name, optimizer in optimizers.items():
        print(f"\nОбучение с оптимизатором: {name}")
        
        # Создание модели
        model = Model()
        model.add(Dense(3, 32, activation=ReLU(), weight_init='he'))
        model.add(Dense(32, 16, activation=ReLU(), weight_init='he'))
        model.add(Dense(16, 1, activation=None))
        model.compile(loss=MSE(), optimizer=optimizer)
        
        # Обучение модели
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_test, y_test),
            verbose=True
        )
        
        # Оценка модели
        y_pred = model.predict(X_test)
        mse = np.mean(np.square(y_pred - y_test))
        
        print(f"MSE на тестовой выборке: {mse:.6f}")
        
        histories[name] = history
        mses[name] = mse
    
    # Визуализация результатов
    plt.figure(figsize=(12, 6))
    
    # График потерь на обучении
    plt.subplot(1, 2, 1)
    for name, history in histories.items():
        plt.plot(history['train_loss'], label=name)
    plt.title('Потери на обучающей выборке')
    plt.xlabel('Эпоха')
    plt.ylabel('MSE')
    plt.legend()
    
    # График потерь на валидации
    plt.subplot(1, 2, 2)
    for name, history in histories.items():
        plt.plot(history['val_loss'], label=f"{name} (MSE={mses[name]:.6f})")
    plt.title('Потери на валидационной выборке')
    plt.xlabel('Эпоха')
    plt.ylabel('MSE')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return histories, mses

# ЗАПУСК ПРИМЕРОВ
if __name__ == "__main__":
    # Выберите пример для запуска:
    example = input("Выберите пример для запуска (1 - MNIST, 2 - Iris, 3 - Сравнение оптимизаторов): ")
    
    if example == "1":
        print("\nЗапуск примера на датасете MNIST...\n")
        mnist_model, mnist_history = run_mnist_example()
    elif example == "2":
        print("\nЗапуск примера на датасете Iris...\n")
        iris_model, iris_history = run_iris_example()
    elif example == "3":
        print("\nЗапуск сравнения оптимизаторов...\n")
        opt_histories, opt_mses = run_optimizer_comparison()
    else:
        print("Неверный выбор. Пожалуйста, выберите 1, 2 или 3.")