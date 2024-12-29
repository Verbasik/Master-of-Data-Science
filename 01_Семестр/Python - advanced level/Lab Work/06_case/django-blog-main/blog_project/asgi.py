# django-blog-main/blog_project/asgi.py
"""
Description:
   Конфигурация ASGI для проекта blog_project.
   
Note:
   Предоставляет ASGI-вызываемый объект как переменную уровня 
   модуля 'application'.

Link:
   https://docs.djangoproject.com/en/5.1/howto/deployment/asgi/
"""

# Стандартные библиотеки
import os

# Django импорты
from django.core.asgi import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'blog_project.settings')

application = get_asgi_application()