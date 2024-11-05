import datetime

def display_current_datetime() -> str:
    """
    Description:
      Отображает текущую дату и время.

    Returns:
        Строка с текущей датой и временем в формате 'YYYY-MM-DD HH:MM:SS'.

    Examples:
        >>> display_current_datetime()
        '2023-10-05 14:30:45'
    """
    current_datetime = datetime.datetime.now()
    return current_datetime.strftime('%Y-%m-%d %H:%M:%S')

def calculate_date_difference(date1: str, date2: str) -> str:
    """
    Description:
      Вычисляет разницу между двумя датами.

    Args:
        date1: Первая дата в формате 'YYYY-MM-DD HH:MM:SS'.
        date2: Вторая дата в формате 'YYYY-MM-DD HH:MM:SS'.

    Returns:
        Строка с разницей между датами в формате 'DD days, HH hours, MM minutes, SS seconds'.

    Raises:
        ValueError: Если формат даты не соответствует 'YYYY-MM-DD HH:MM:SS'.

    Examples:
        >>> calculate_date_difference('2023-10-01 12:00:00', '2023-10-05 14:30:45')
        '4 days, 2 hours, 30 minutes, 45 seconds'
    """
    try:
        datetime1 = datetime.datetime.strptime(date1, '%Y-%m-%d %H:%M:%S')
        datetime2 = datetime.datetime.strptime(date2, '%Y-%m-%d %H:%M:%S')
    except ValueError as e:
        raise ValueError("Неверный формат даты. Используйте формат 'YYYY-MM-DD HH:MM:SS'.") from e

    difference = datetime2 - datetime1
    days = difference.days
    seconds = difference.seconds
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    return f'{days} days, {hours} hours, {minutes} minutes, {seconds} seconds'

def convert_string_to_datetime(date_str: str) -> datetime.datetime:
    """
    Description:
      Преобразует строку в объект datetime.

    Args:
        date_str: Дата в строковом формате 'YYYY-MM-DD HH:MM:SS'.

    Returns:
        Объект datetime.

    Raises:
        ValueError: Если формат даты не соответствует 'YYYY-MM-DD HH:MM:SS'.

    Examples:
        >>> convert_string_to_datetime('2023-10-05 14:30:45')
        datetime.datetime(2023, 10, 5, 14, 30, 45)
    """
    try:
        return datetime.datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
    except ValueError as e:
        raise ValueError("Неверный формат даты. Используйте формат 'YYYY-MM-DD HH:MM:SS'.") from e

# Пример использования функций
if __name__ == "__main__":
    print("Текущая дата и время:", display_current_datetime())
    print("Разница между датами:", calculate_date_difference('2023-10-01 12:00:00', '2023-10-05 14:30:45'))
    print("Преобразование строки в datetime:", convert_string_to_datetime('2023-10-05 14:30:45'))