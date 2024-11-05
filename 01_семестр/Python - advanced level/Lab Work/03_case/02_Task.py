import string
from collections import Counter

def count_unique_words(text: str) -> int:
    """
    Description:
      Функция для подсчета количества уникальных слов в строке, игнорируя знаки препинания и пробелы.

    Args:
        text: Входная строка, в которой нужно подсчитать уникальные слова.

    Returns:
        Количество уникальных слов в строке.

    Examples:
        >>> count_unique_words("Привет, мир! Привет, Python.")
        2
    """
    if not text:
        return 0
    
    # Удаляем знаки препинания и приводим строку к нижнему регистру
    translator = str.maketrans('', '', string.punctuation)
    cleaned_text = text.translate(translator).lower()
    
    # Разбиваем строку на слова
    words = cleaned_text.split()
    
    # Подсчитываем уникальные слова с помощью Counter
    word_counts = Counter(words)
    
    # Возвращаем количество уникальных слов
    return len(word_counts)

# Пример использования
print(count_unique_words("Привет, мир! Привет, Python."))  # Вывод: 3