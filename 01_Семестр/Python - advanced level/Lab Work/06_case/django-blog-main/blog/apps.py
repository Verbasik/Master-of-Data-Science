# django-blog-main/blog/apps.py

from django.apps import AppConfig

class BlogConfig(AppConfig):
    """
    Description:
        Конфигурация приложения для блога в Django.

    Args:
        default_auto_field: Тип поля по умолчанию для автоинкрементных значений.
        name: Имя приложения в Django проекте.

    Returns:
        None

    Examples:
        Пример использования конфигурации для регистрации приложения 'blog'.
    """
    
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'blog'
