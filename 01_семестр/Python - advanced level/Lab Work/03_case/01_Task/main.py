# main.py
from math_operations.sum_calculator import SumCalculator

if __name__ == "__main__":
    # Создаем экземпляр класса SumCalculator
    calculator = SumCalculator()
    # Вычисляем сумму чисел в списке
    result = calculator.calculate_sum([1, 2, 3, 4])
    # Выводим результат
    print(f"Сумма чисел: {result}")