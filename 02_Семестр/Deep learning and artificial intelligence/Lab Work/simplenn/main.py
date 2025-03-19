"""
Главный скрипт для запуска примеров использования фреймворка SimpleNN.
"""

import sys
import os

# Добавляем корневую директорию проекта в путь для корректного импорта
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from examples.mnist_example import run_mnist_example
from examples.iris_example import run_iris_example
from examples.regression_example import run_regression_example
from examples.quick_start import run_quick_start_example

def print_menu():
    """Вывод меню с доступными примерами"""
    print("\n" + "=" * 50)
    print("SimpleNN - примеры использования")
    print("=" * 50)
    print("1. Классификация рукописных цифр (MNIST)")
    print("2. Классификация цветков Iris")
    print("3. Регрессия на синтетических данных")
    print("4. Быстрый старт (бинарная классификация)")
    print("0. Выход")
    print("=" * 50)
    return input("Выберите пример для запуска (0-4): ")

def main():
    """Основная функция для запуска примеров"""
    while True:
        choice = print_menu()
        
        if choice == "0":
            print("Выход из программы")
            break
        elif choice == "1":
            print("\nЗапуск примера MNIST...\n")
            model, history = run_mnist_example()
        elif choice == "2":
            print("\nЗапуск примера Iris...\n")
            model, history = run_iris_example()
        elif choice == "3":
            print("\nЗапуск примера регрессии...\n")
            model, history = run_regression_example()
        elif choice == "4":
            print("\nЗапуск быстрого примера...\n")
            model, history = run_quick_start_example()
        else:
            print("Неверный выбор. Пожалуйста, выберите 0-4.")
        
        input("\nНажмите Enter для продолжения...")

if __name__ == "__main__":
    main()