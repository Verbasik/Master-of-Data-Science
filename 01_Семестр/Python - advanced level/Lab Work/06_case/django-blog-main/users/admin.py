# django-blog-main/users/admin.py
# Импорты Django
from django.contrib import admin

# Импорты локальных моделей
from .models import Profile

# Регистрация модели Profile в админке Django
admin.site.register(Profile)