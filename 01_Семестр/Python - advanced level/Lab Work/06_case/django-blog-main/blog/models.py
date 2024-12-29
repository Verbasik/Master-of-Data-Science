# django-blog-main/blog/models.py
# Стандартные библиотеки Django
from django.contrib.auth.models import User
from django.db import models
from django.urls import reverse
from django.utils import timezone


class Post(models.Model):
    """
    Description:
        Модель поста в блоге.

    Args:
        models.Model: Базовый класс для моделей Django

    Attributes:
        title (CharField): Заголовок поста, максимум 75 символов
        subtitle (CharField): Подзаголовок, максимум 100 символов
        content (TextField): Содержимое поста
        date_posted (DateTimeField): Дата публикации
        author (ForeignKey): Ссылка на автора поста
    """
    title = models.CharField(max_length=75)
    subtitle = models.CharField(max_length=100)
    content = models.TextField()
    date_posted = models.DateTimeField(default=timezone.now)
    author = models.ForeignKey(
        User,
        on_delete=models.CASCADE
    )

    def __str__(self) -> str:
        """
        Description:
            Строковое представление модели.

        Returns:
            str: Заголовок поста
        """
        return self.title

    def get_absolute_url(self) -> str:
        """
        Description:
            Получает абсолютный URL поста.

        Returns:
            str: URL страницы детального просмотра поста
        """
        return reverse('post-detail', kwargs={'pk': self.pk})

    def total_likes(self) -> int:
        """
        Description:
            Подсчитывает общее количество лайков поста.

        Returns:
            int: Количество лайков
        """
        return self.likes.count()


class Like(models.Model):
    """
    Description:
        Модель лайка для поста.

    Args:
        models.Model: Базовый класс для моделей Django

    Attributes:
        user (ForeignKey): Пользователь, поставивший лайк
        post (ForeignKey): Пост, которому поставлен лайк
        created (DateTimeField): Дата создания лайка
    """
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    post = models.ForeignKey(
        Post,
        related_name='likes',
        on_delete=models.CASCADE
    )
    created = models.DateTimeField(auto_now_add=True)

    class Meta:
        """
        Description:
            Метаданные модели, определяющие уникальность лайков.
        """
        unique_together = ('user', 'post')

    def __str__(self) -> str:
        """
        Description:
            Строковое представление модели.

        Returns:
            str: Информация о лайке в формате 'пользователь likes пост'
        """
        return f'{self.user.username} likes {self.post.title}'