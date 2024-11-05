def read_numeric_lines(filename: str) -> None:
    """
    Description:
        Читает текстовый файл и выводит на экран только те строки, которые содержат числовые значения.

    Args:
        filename (str): Имя файла для чтения.

    Raises:
        FileNotFoundError: Если файл не найден.
        TypeError: Если внутри файла попадется значение, отличное от числа.
    """
    try:
        with open(filename, 'r') as file:
            for line in file:
                # Проверяем, содержит ли строка числовые значения
                if any(char.isdigit() for char in line):
                    try:
                        # Пробуем преобразовать строку в число
                        float(line.strip())
                        print(line.strip())
                    except ValueError:
                        # Если строка не может быть преобразована в число, пропускаем её
                        continue
    except FileNotFoundError:
        print(f"Файл '{filename}' не найден.")

# Пример использования
read_numeric_lines("example.txt")