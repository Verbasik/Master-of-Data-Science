class Animal:
    """
    Description:
        Класс, представляющий животное.

    Attributes:
        name (str): Имя животного.
        sound (str): Звук, который издает животное.
    """
    def __init__(self, name: str, sound: str):
        """
        Description:
            Инициализация объекта Animal.

        Args:
            name (str): Имя животного.
            sound (str): Звук, который издает животное.
        """
        self.name = name
        self.sound = sound

    def makesound(self) -> None:
        """
        Description:
            Выводит на экран звук, который издает животное.
        """
        print(f"{self.name} говорит: {self.sound}")

class Cat(Animal):
    """
    Description:
        Подкласс, представляющий кошку.

    Attributes:
        name (str): Имя кошки.
        sound (str): Звук, который издает кошка.
        color (str): Цвет кошки.
    """
    def __init__(self, name: str, sound: str, color: str):
        """
        Description:
            Инициализация объекта Cat.

        Args:
            name (str): Имя кошки.
            sound (str): Звук, который издает кошка.
            color (str): Цвет кошки.
        """
        super().__init__(name, sound)
        self.color = color

    def makesound(self) -> None:
        """
        Description:
            Выводит на экран звук, который издает кошка, с указанием цвета.
        """
        super().makesound()
        print(f"Цвет кошки: {self.color}")

class Dog(Animal):
    """
    Description:
        Подкласс, представляющий собаку.

    Attributes:
        name (str): Имя собаки.
        sound (str): Звук, который издает собака.
        color (str): Цвет собаки.
    """
    def __init__(self, name: str, sound: str, color: str):
        """
        Description:
            Инициализация объекта Dog.

        Args:
            name (str): Имя собаки.
            sound (str): Звук, который издает собака.
            color (str): Цвет собаки.
        """
        super().__init__(name, sound)
        self.color = color

    def makesound(self) -> None:
        """
        Description:
            Выводит на экран звук, который издает собака, с указанием цвета.
        """
        super().makesound()
        print(f"Цвет собаки: {self.color}")

# Пример использования
cat = Cat("Мурка", "Мяу", "Серый")
dog = Dog("Бобик", "Гав", "Черный")

cat.makesound()  # Вывод: Мурка говорит: Мяу
                 #         Цвет кошки: Серый
dog.makesound()  # Вывод: Бобик говорит: Гав
                 #         Цвет собаки: Черный