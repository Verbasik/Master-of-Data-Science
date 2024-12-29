# django-blog-main/users/signals.py
# Импорты Django
from django.db.models.signals import post_save
from django.contrib.auth.models import User
from django.dispatch import receiver

# Импорты локальных моделей
from .models import Profile


@receiver(post_save, sender=User)
def create_profile(sender, instance: User, created: bool, **kwargs) -> None:
    """
    Description:
        Сигнал, который создает профиль пользователя при создании нового пользователя.

    Args:
        sender: Модель, отправившая сигнал (User).
        instance: Экземпляр модели User.
        created: Флаг, указывающий, был ли объект создан.
        **kwargs: Дополнительные аргументы.
    """
    if created:
        Profile.objects.create(user=instance)


@receiver(post_save, sender=User)
def save_profile(sender, instance: User, **kwargs) -> None:
    """
    Description:
        Сигнал, который сохраняет профиль пользователя при сохранении пользователя.

    Args:
        sender: Модель, отправившая сигнал (User).
        instance: Экземпляр модели User.
        **kwargs: Дополнительные аргументы.
    """
    instance.profile.save()