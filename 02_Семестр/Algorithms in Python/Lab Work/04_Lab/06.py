from typing import List

def solve() -> None:
    """
    Description:
    ---------------
        Находит минимальное количество песен, которые нужно прослушать,
        чтобы набрать не менее p минут, начиная с определённой песни.

    Args:
    ---------------
        None (ввод осуществляется через стандартный ввод)

    Returns:
    ---------------
        None (вывод осуществляется через стандартный вывод)

    Examples:
    ---------------
        >>> solve()
        Ввод:
        3 10
        3 4 5
        Вывод:
        1 3
    """
    n, p = map(int, input().split())
    a = list(map(int, input().split()))

    playlist_sum = sum(a)

    min_songs = float('inf')
    best_start_song = -1

    # Проходим по всем возможным стартовым индексам
    for start_index in range(n):
        current_sum = 0
        songs_count = 0

        # Если суммарная длина плейлиста больше 0
        if playlist_sum > 0:
            full_cycles = 0

            # Вычисляем количество полных циклов
            if p > playlist_sum:
                full_cycles = (p - 1) // playlist_sum
                current_sum += full_cycles * playlist_sum
                songs_count += full_cycles * n

            remaining_p = p - current_sum

            temp_songs_count = 0
            temp_sum = 0

            # Проходим по плейлисту, начиная с текущего стартового индекса
            for _ in range(n):
                current_song_index = (start_index + temp_songs_count) % n
                temp_sum += a[current_song_index]
                temp_songs_count += 1

                # Если набрали достаточно минут
                if temp_sum >= remaining_p:
                    songs_count += temp_songs_count
                    current_sum += temp_sum
                    break

        # Если текущая сумма больше или равна p, обновляем минимальное количество песен
        if current_sum >= p:
            if songs_count < min_songs:
                min_songs = songs_count
                best_start_song = start_index + 1

    print(best_start_song, min_songs)

if __name__ == "__main__":
    solve()
