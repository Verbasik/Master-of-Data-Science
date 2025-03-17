# SimpleNN - Простой нейросетевой фреймворк на Python

SimpleNN - это легковесный нейросетевой фреймворк, написанный на чистом Python с использованием только библиотек NumPy и Pandas. Фреймворк предоставляет интуитивно понятный API для создания, обучения и оценки полносвязных нейронных сетей.

## Особенности

- **Простота использования**: создание и обучение нейросетей "в несколько строк"
- **Архитектурная гибкость**: возможность создания многослойных нейросетей с различными активациями
- **Разнообразные оптимизаторы**: SGD, Momentum SGD, RMSprop, Adam и Gradient Clipping
- **Функции активации**: ReLU, Sigmoid, Tanh, LeakyReLU, Softmax
- **Функции потерь**: MSE, MAE, Binary Cross-Entropy, Categorical Cross-Entropy
- **Инструменты для работы с данными**: нормализация, стандартизация, one-hot кодирование, разбиение данных
- **Регуляризация**: Dropout, Batch Normalization
- **Визуализация**: графики обучения, визуализация результатов

## Установка

Для использования фреймворка требуются следующие зависимости:
- Python 3.6+
- NumPy
- Pandas
- Matplotlib
- tqdm (для прогресс-баров)

Установка зависимостей:

```bash
pip install numpy pandas matplotlib tqdm
```

## Быстрый старт

Вот пример создания и обучения простой нейронной сети для задачи классификации:

```python
import numpy as np
from simplenn import *

# Создаем синтетический датасет
X = np.random.randn(500, 2)
y = (X[:, 0]**2 + X[:, 1]**2 < 1).astype(int).reshape(-1, 1)

# Разделяем на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = DataUtils.train_test_split(X, y, test_size=0.2)

# Создаем модель в несколько строк
model = Model()
model.add(Dense(2, 16, activation=ReLU()))
model.add(Dense(16, 8, activation=ReLU()))
model.add(Dense(8, 1, activation=Sigmoid()))
model.compile(loss=BinaryCrossEntropy(), optimizer=Adam())

# Обучаем модель
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Оцениваем модель
y_pred = model.predict(X_test)
accuracy = model._calculate_accuracy(y_pred, y_test)
print(f"Точность: {accuracy:.4f}")
```

## Основные компоненты

### Слои

- `Dense`: полносвязный слой
- `Dropout`: слой для регуляризации
- `BatchNormalization`: слой для нормализации данных

### Функции активации

- `ReLU`: Rectified Linear Unit
- `LeakyReLU`: Leaky Rectified Linear Unit
- `Sigmoid`: Сигмоидная функция
- `Tanh`: Гиперболический тангенс
- `Softmax`: Функция для многоклассовой классификации

### Функции потерь

- `MSE`: Среднеквадратичная ошибка (Mean Squared Error)
- `MAE`: Средняя абсолютная ошибка (Mean Absolute Error)
- `BinaryCrossEntropy`: Бинарная кросс-энтропия
- `CategoricalCrossEntropy`: Категориальная кросс-энтропия

### Оптимизаторы

- `SGD`: Стохастический градиентный спуск
- `MomentumSGD`: SGD с моментом
- `RMSprop`: Root Mean Square Propagation
- `Adam`: Adaptive Moment Estimation
- `GradientClipping`: Отсечение градиентов (обертка для других оптимизаторов)

### Утилиты для работы с данными

- `normalize`: Нормализация данных (от 0 до 1)
- `standardize`: Стандартизация данных (среднее = 0, ст. отклонение = 1)
- `one_hot_encode`: Преобразование меток классов в one-hot формат
- `train_test_split`: Разделение данных на обучающую и тестовую выборки
- `batch_generator`: Генератор батчей для обучения
- `k_fold_split`: Разделение данных на k разбиений для кросс-валидации
- `data_augmentation`: Аугментация данных

## Примеры использования

### MNIST

```python
from sklearn.datasets import fetch_openml
from simplenn import *

# Загрузка данных MNIST
mnist = fetch_openml('mnist_784', version=1, parser='auto')
X = mnist.data.astype(np.float32).values
y = mnist.target.astype(np.int32).values

# Предобработка данных
X = DataUtils.normalize(X)
y_one_hot = DataUtils.one_hot_encode(y, 10)

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = DataUtils.train_test_split(X, y_one_hot, test_size=0.2)

# Создание модели
model = Model()
model.add(Dense(784, 128, activation=ReLU(), weight_init='he'))
model.add(BatchNormalization(128))
model.add(Dropout(0.3))
model.add(Dense(128, 64, activation=ReLU(), weight_init='he'))
model.add(Dense(64, 10, activation=Softmax(), weight_init='he'))
model.compile(loss=CategoricalCrossEntropy(), optimizer=Adam(learning_rate=0.001))

# Обучение модели
model.fit(X_train, y_train, epochs=10, batch_size=128, validation_data=(X_test, y_test))
```

### Iris

```python
from sklearn.datasets import load_iris
from simplenn import *

# Загрузка данных Iris
iris = load_iris()
X = iris.data.astype(np.float32)
y = iris.target

# Предобработка данных
X = DataUtils.standardize(X)
y_one_hot = DataUtils.one_hot_encode(y, 3)

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = DataUtils.train_test_split(X, y_one_hot, test_size=0.2)

# Создание модели
model = Model()
model.add(Dense(4, 10, activation=ReLU(), weight_init='he'))
model.add(Dense(10, 3, activation=Softmax(), weight_init='he'))
model.compile(loss=CategoricalCrossEntropy(), optimizer=MomentumSGD(learning_rate=0.01, momentum=0.9))

# Обучение модели
model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test))
```

## Ранняя остановка обучения

```python
def early_stopping_callback(info, patience=5, min_delta=0.001):
    """Колбэк для ранней остановки обучения"""
    if not hasattr(early_stopping_callback, 'best_val_loss'):
        early_stopping_callback.best_val_loss = float('inf')
        early_stopping_callback.wait = 0
        early_stopping_callback.stop_training = False
    
    current_val_loss = info['val_loss']
    
    if current_val_loss < early_stopping_callback.best_val_loss - min_delta:
        early_stopping_callback.best_val_loss = current_val_loss
        early_stopping_callback.wait = 0
    else:
        early_stopping_callback.wait += 1
        if early_stopping_callback.wait >= patience:
            early_stopping_callback.stop_training = True
            print(f"\nРанняя остановка на эпохе {info['epoch']}")
            return True
    
    return False

# Использование колбэка
def callback_wrapper(info):
    should_stop = early_stopping_callback(info, patience=3)
    if should_stop:
        model.train_interrupted = True

model.train_interrupted = False
model.fit(X_train, y_train, epochs=100, batch_size=32, 
          validation_data=(X_test, y_test), callback=callback_wrapper)
```

## Сохранение и загрузка весов

```python
# Сохранение весов модели
model.save_weights('model_weights.npz')

# Загрузка весов модели
model.load_weights('model_weights.npz')
```

## Сводка о модели и визуализация

```python
# Вывод сводки о модели
model.summary()

# Построение графиков обучения
model.plot_history()
```

## Кросс-валидация

```python
# Кросс-валидация с 5 разбиениями
n_folds = 5
folds = DataUtils.k_fold_split(X, y, n_folds=n_folds)

val_accuracies = []

for fold_idx, (X_train, X_val, y_train, y_val) in enumerate(folds):
    # Создание и обучение модели для каждого разбиения
    model = create_model()  # Функция для создания модели
    model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val))
    
    # Оценка модели
    y_pred = model.predict(X_val)
    accuracy = model._calculate_accuracy(y_pred, y_val)
    val_accuracies.append(accuracy)

# Вычисление средней точности
mean_accuracy = np.mean(val_accuracies)
```

## Лицензия

Этот проект распространяется под лицензией MIT.