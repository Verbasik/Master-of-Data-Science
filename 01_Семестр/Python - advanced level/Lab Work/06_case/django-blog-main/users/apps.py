# django-blog-main/users/apps.py
# Импорты Django
from django.apps import AppConfig


class UsersConfig(AppConfig):
    """
    Description:
        Класс конфигурации приложения 'users'.

    Attributes:
        default_auto_field: Тип поля для автоматического создания первичного ключа.
        name: Имя приложения.
    """

    default_auto_field = 'django.db.models.BigAutoField'
    name = 'users'

    def ready(self) -> None:
        """
        Description:
            Метод, вызываемый при готовности приложения.
            Импортирует сигналы для их регистрации.

        Examples:
            Сигналы из модуля 'users.signals' будут зарегистрированы автоматически.
        """
        import users.signals