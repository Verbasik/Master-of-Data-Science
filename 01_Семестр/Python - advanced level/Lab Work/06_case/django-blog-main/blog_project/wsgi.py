# django-blog-main/blog_project/wsgi.py
"""
Description:
   Конфигурация WSGI для проекта blog_project.
   
Note:
   Предоставляет WSGI-вызываемый объект как переменную уровня модуля 'application'.

Link: 
   https://docs.djangoproject.com/en/5.1/howto/deployment/wsgi/
"""

# Стандартные библиотеки
import os

# Django импорты
from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'blog_project.settings')

application = get_wsgi_application()