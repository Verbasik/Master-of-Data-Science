# django-blog-main/users/models.py
# Импорты стандартной библиотеки
import random

# Импорты Django
from django.db import models
from django.contrib.auth.models import User

# Импорты сторонних библиотек
from PIL import Image


class Profile(models.Model):
    """
    Description:
        Модель профиля пользователя. Содержит информацию о пользователе,
        его изображении и биографии.

    Attributes:
        user: Связь один-к-одному с моделью User.
        image: Поле для загрузки изображения профиля.
        bio: Поле для краткой биографии пользователя.
    """

    user = models.OneToOneField(User, on_delete=models.CASCADE)
    image = models.ImageField(upload_to='profile_pics')
    bio = models.CharField(max_length=40, blank=True, null=True)

    def save(self, *args, **kwargs) -> None:
        """
        Description:
            Переопределенный метод сохранения объекта Profile.
            Устанавливает случайное изображение по умолчанию и биографию,
            если они не указаны. Также уменьшает размер изображения,
            если оно превышает 300x300 пикселей.

        Args:
            *args: Позиционные аргументы.
            **kwargs: Именованные аргументы.
        """
        if not self.pk:
            default_images = [
                'default.jpg',
                'default2.jpg',
                'default3.jpg',
                'default4.jpg',
            ]

            # Выбор случайного изображения, если изображение не указано
            if not self.image:
                chosen_image = random.choice(default_images)
                self.image.name = chosen_image

            # Установка биографии по умолчанию, если она не указана
            if not self.bio:
                self.bio = "Hello, there!"

        super().save(*args, **kwargs)

        # Уменьшение размера изображения, если оно больше 300x300 пикселей
        if self.image:
            img_path = self.image.path
            img = Image.open(img_path)
            if img.height > 300 or img.width > 300:
                output_size = (300, 300)
                img.thumbnail(output_size)
                img.save(img_path)

    def __str__(self) -> str:
        """
        Description:
            Возвращает строковое представление объекта Profile.

        Returns:
            Строка в формате '{username} Profile'.
        """
        return f'{self.user.username} Profile'