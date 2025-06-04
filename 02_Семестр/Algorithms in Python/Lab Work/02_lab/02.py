import sys
from typing import List

def solve() -> None:
    """
    Description:
        Основная функция для решения задачи.
        Читает входные данные, проверяет возможность получения строки p из строки t
        после удаления первых k символов по заданному порядку.

    Args:
        None

    Returns:
        None

    Raises:
        None

    Examples:
        >>> solve()
        # Ввод: 3 ab 1 2 3
        # Вывод: 1
    """
    input_data = sys.stdin.read().split()
    if not input_data:
        return

    t = input_data[0]
    p = input_data[1]
    n = len(t)
    a = list(map(int, input_data[2:]))

    def possible(k: int) -> bool:
        """
        Description:
            Проверяет, можно ли получить слово p после удаления первых k символов
            по заданному порядку.

        Args:
            k: Количество удаляемых символов

        Returns:
            True, если можно получить слово p, иначе False

        Raises:
            None

        Examples:
            >>> possible(1)
            True
        """
        removed = [False] * (n + 1)  # Используем индексацию с 1 до n
        for i in range(k):
            removed[a[i]] = True

        j = 0  # указатель на символы строки p
        # Проходим по строке t, пропуская удалённые символы
        for i in range(1, n + 1):
            if not removed[i]:
                if j < len(p) and t[i - 1] == p[j]:
                    j += 1
                    if j == len(p):
                        break

        return j == len(p)

    low, high = 0, n
    ans = 0
    # Бинарный поиск (метод дихотомии) по количеству удаляемых символов
    while low <= high:
        mid = (low + high) // 2
        if possible(mid):
            ans = mid
            low = mid + 1
        else:
            high = mid - 1

    sys.stdout.write(str(ans))

if __name__ == '__main__':
    solve()
