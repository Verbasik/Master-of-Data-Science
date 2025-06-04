# Dockerfile

# Базовый образ
FROM python:3.10-slim as builder

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Установка рабочей директории
WORKDIR /app

# Копирование только requirements.txt из правильной директории
COPY django-blog-main/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Финальный образ
FROM python:3.10-slim

# Копирование установленных пакетов
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

# Установка рабочей директории
WORKDIR /app

# Копирование только нужной директории проекта
COPY django-blog-main/ .

# Создание необходимых директорий
RUN mkdir -p media static

# Создание и настройка пользователя
RUN useradd -m django
RUN chown -R django:django /app
USER django

# Проверка здоровья
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Открытие порта
EXPOSE 8000

# Запуск приложения
CMD ["sh", "-c", "python manage.py migrate && python manage.py runserver 0.0.0.0:8000"]