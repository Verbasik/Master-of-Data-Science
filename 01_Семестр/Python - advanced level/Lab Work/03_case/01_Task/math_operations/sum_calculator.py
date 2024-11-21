# math_operations/sum_calculator.py
class SumCalculator:
    """
    Description:
        Класс для вычисления суммы чисел в списке.
    """

    def calculate_sum(self, numbers: list[int]) -> int:
        """
        Вычисляет сумму всех чисел в переданном списке.

        Args:
            numbers: Список целых чисел.

        Returns:
            Сумма чисел в списке.

        Examples:
            >>> calculator = SumCalculator()
            >>> calculator.calculate_sum([1, 2, 3, 4])
            10
        """
        # Используем встроенную функцию sum для вычисления суммы
        return sum(numbers)