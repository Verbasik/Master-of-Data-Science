# Инструкция по установке и запуску SimpleNN

## Установка

### Вариант 1: Установка из исходного кода

1. Клонировать репозиторий:
```bash
git clone https://github.com/username/simplenn.git
cd simplenn
```

2. Установить в режиме разработки:
```bash
pip install -e .
```

### Вариант 2: Использование без установки

1. Просто скопируйте структуру директорий и убедитесь, что все зависимости установлены:
```bash
pip install numpy pandas matplotlib tqdm
```

## Структура проекта

```
simplenn/
├── __init__.py                # Основной файл инициализации пакета
├── core/                      # Основные компоненты фреймворка
│   ├── __init__.py
│   ├── tensor.py              # Класс Tensor
│   ├── layer.py               # Базовые классы слоев
│   ├── activation.py          # Функции активации
│   ├── loss.py                # Функции потерь
│   ├── optimizer.py           # Оптимизаторы
│   └── model.py               # Основной класс Model
├── utils/                     # Утилиты
│   ├── __init__.py
│   └── data_utils.py          # Функции для работы с данными
└── examples/                  # Примеры использования
    ├── __init__.py
    ├── mnist_example.py       # Пример с MNIST
    ├── iris_example.py        # Пример с Iris
    ├── regression_example.py  # Пример регрессии
    └── quick_start.py         # Пример быстрого старта
```

## Запуск примеров

### Запуск из основного скрипта

```bash
python main.py
```

Этот скрипт откроет интерактивное меню для выбора примера.

### Запуск отдельных примеров

```bash
# Пример MNIST
python -m simplenn.examples.mnist_example

# Пример Iris
python -m simplenn.examples.iris_example

# Пример регрессии
python -m simplenn.examples.regression_example

# Быстрый старт
python -m simplenn.examples.quick_start
```

## Зависимости

- Python 3.6+
- NumPy
- Pandas
- Matplotlib
- tqdm

## Примечание

Для запуска примера MNIST потребуется дополнительно установить scikit-learn:

```bash
pip install scikit-learn
```