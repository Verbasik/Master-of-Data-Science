# django-blog-main/blog_project/settings.py
"""
Description:
   Настройки Django для проекта blog_project.

Links:
   - Settings: https://docs.djangoproject.com/en/5.1/topics/settings/
   - Settings Reference: https://docs.djangoproject.com/en/5.1/ref/settings/
"""

# Стандартные библиотеки
import os
from pathlib import Path

# Базовые настройки проекта
BASE_DIR = Path(__file__).resolve().parent.parent
SECRET_KEY = 'django-insecure-e84#3=dytx+d+fp+_b9*8#6ehsrxsg&lfllgw$a6t0)w&btauj'
DEBUG = True
ALLOWED_HOSTS = []

# Конфигурация приложений
INSTALLED_APPS = [
   # Пользовательские приложения
   'blog.apps.BlogConfig',
   'users.apps.UsersConfig',

   # Стандартные приложения Django
   'django.contrib.admin',
   'django.contrib.auth',
   'django.contrib.contenttypes',
   'django.contrib.sessions',
   'django.contrib.messages',
   'django.contrib.staticfiles',

   # Сторонние приложения
   'crispy_forms',
   'crispy_bootstrap4',
]

# Настройки Crispy Forms
CRISPY_ALLOWED_TEMPLATE_PACKS = "bootstrap4"
CRISPY_TEMPLATE_PACK = 'bootstrap4'

# Middleware
MIDDLEWARE = [
   'django.middleware.security.SecurityMiddleware',
   'django.contrib.sessions.middleware.SessionMiddleware',
   'django.middleware.common.CommonMiddleware',
   'django.middleware.csrf.CsrfViewMiddleware',
   'django.contrib.auth.middleware.AuthenticationMiddleware',
   'django.contrib.messages.middleware.MessageMiddleware',
   'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

# Конфигурация URL и шаблонов
ROOT_URLCONF = 'blog_project.urls'

TEMPLATES = [
   {
       'BACKEND': 'django.template.backends.django.DjangoTemplates',
       'DIRS': [],
       'APP_DIRS': True,
       'OPTIONS': {
           'context_processors': [
               'django.template.context_processors.debug',
               'django.template.context_processors.request',
               'django.contrib.auth.context_processors.auth',
               'django.contrib.messages.context_processors.messages',
           ],
       },
   },
]

WSGI_APPLICATION = 'blog_project.wsgi.application'

# Настройки базы данных
DATABASES = {
   'default': {
       'ENGINE': 'django.db.backends.sqlite3',
       'NAME': BASE_DIR / 'db.sqlite3',
   }
}

# Валидация паролей
AUTH_PASSWORD_VALIDATORS = [
   {
       'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
   },
   {
       'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
   },
   {
       'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
   },
   {
       'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
   },
]

# Интернационализация
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

# Статические и медиа файлы
STATIC_URL = '/static/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')
MEDIA_URL = '/media/'

# Настройки авторизации
LOGIN_REDIRECT_URL = 'blog-home'
LOGIN_URL = "login"

# Прочие настройки
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'