# django-blog-main/blog/admin.py

# Импорты стандартных библиотек
from django.contrib import admin

# Импорты сторонних библиотек
from .models import Post

# Регистрация модели Post в административной панели Django
admin.site.register(Post)
