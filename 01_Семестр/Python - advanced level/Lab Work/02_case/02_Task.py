class DataBuffer:
    """
    Description:
        Класс, представляющий буфер данных.

    Attributes:
        buffer (list): Список для хранения данных в буфере.
    """
    def __init__(self):
        """
        Description:
            Инициализация объекта DataBuffer.
        """
        self.buffer = []

    def add_data(self, data):
        """
        Description:
            Добавляет данные в буфер. Если в буфере уже есть хотя бы 5 элементов,
            выводит сообщение о переполнении буфера и очищает его.

        Args:
            data: Данные для добавления в буфер.
        """
        self.buffer.append(data)
        if len(self.buffer) >= 5:
            print("Буфер переполнен. Очистка буфера.")
            self.buffer.clear()

    def get_data(self):
        """
        Description:
            Получает данные из буфера. Если буфер пуст, выводит сообщение об отсутствии данных.
            После получения данных буфер очищается.

        Returns:
            list: Данные из буфера.
        """
        if not self.buffer:
            print("Буфер пуст. Данные отсутствуют.")
            return []
        data = self.buffer.copy()
        self.buffer.clear()
        return data

# Пример использования
buffer = DataBuffer()
buffer.add_data("data1")
buffer.add_data("data2")
buffer.add_data("data3")
buffer.add_data("data4")
buffer.add_data("data5")  # Буфер переполнен. Очистка буфера.

print(buffer.get_data())  # Буфер пуст. Данные отсутствуют.