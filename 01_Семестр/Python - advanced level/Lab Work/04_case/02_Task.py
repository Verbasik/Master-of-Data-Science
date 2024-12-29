import itertools
from typing import Callable, Iterable, Tuple

def infinite_number_generator(start: int = 1, step: int = 1) -> itertools.count:
    """
    Description:
      Создает бесконечный генератор чисел, начиная с заданного значения и с заданным шагом.

    Args:
        start: Начальное значение генератора.
        step: Шаг генератора.

    Returns:
        Бесконечный генератор чисел.

    Raises:
        TypeError: Если тип аргумента start или step не int.
        Exception: Если генератор не может быть создан.
    """
    if not isinstance(start, int) or not isinstance(step, int):
        raise TypeError("Аргументы start и step должны быть типа int.")
    try:
        return itertools.count(start=start, step=step)
    except Exception as e:
        raise Exception(f"Не удалось создать генератор чисел: {e}")

def apply_function_to_iterator(func: Callable, iterator: Iterable[Tuple]) -> itertools.starmap:
    """
    Description:
      Применяет функцию к каждому элементу в итераторе.

    Args:
        func: Функция, которая будет применена к каждому элементу.
        iterator: Итератор, содержащий элементы для обработки.

    Returns:
        Итератор, содержащий результаты применения функции.

    Raises:
        TypeError: Если тип аргумента func не Callable или тип аргумента iterator не Iterable.
        Exception: Если итератор пуст или функция не может быть применена.
    """
    if not callable(func):
        raise TypeError("Аргумент func должен быть функцией.")
    if not isinstance(iterator, Iterable):
        raise TypeError("Аргумент iterator должен быть итерируемым объектом.")
    try:
        return itertools.starmap(func, iterator)
    except Exception as e:
        raise Exception(f"Не удалось применить функцию к итератору: {e}")

def combine_iterators(*iterators: Iterable) -> itertools.chain:
    """
    Description:
      Объединяет несколько итераторов в один.

    Args:
        *iterators: Итераторы, которые будут объединены.

    Returns:
        Объединенный итератор.

    Raises:
        TypeError: Если тип аргумента iterators не Iterable.
        Exception: Если итераторы пусты или не могут быть объединены.
    """
    for iterator in iterators:
        if not isinstance(iterator, Iterable):
            raise TypeError("Аргументы должны быть итерируемыми объектами.")
    try:
        return itertools.chain(*iterators)
    except Exception as e:
        raise Exception(f"Не удалось объединить итераторы: {e}")
    
# Тесты для infinite_number_generator
try:
    gen = infinite_number_generator()
    print("Бесконечный генератор чисел:")
    for _ in range(5):
        print(next(gen))
except Exception as e:
    print(f"Ошибка при создании генератора: {e}")

# Тесты для apply_function_to_iterator
try:
    iterator = [(1, 2), (3, 4), (5, 6)]
    func = lambda x, y: x + y
    result = apply_function_to_iterator(func, iterator)
    print("\nРезультат применения функции к итератору:")
    for item in result:
        print(item)
except Exception as e:
    print(f"Ошибка при применении функции к итератору: {e}")

# Тесты для combine_iterators
try:
    iter1 = [1, 2, 3]
    iter2 = [4, 5, 6]
    combined = combine_iterators(iter1, iter2)
    print("\nРезультат объединения итераторов:")
    for item in combined:
        print(item)
except Exception as e:
    print(f"Ошибка при объединении итераторов: {e}")