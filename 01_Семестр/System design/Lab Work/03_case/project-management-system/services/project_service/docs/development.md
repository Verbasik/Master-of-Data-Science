# Project Service Documentation

## Оглавление
1. [Обзор системы](#обзор-системы)
2. [Архитектура](#архитектура)
3. [API Reference](#api-reference)
4. [Модели данных](#модели-данных)
5. [Аутентификация и безопасность](#аутентификация-и-безопасность)
6. [Взаимодействие с другими сервисами](#взаимодействие-с-другими-сервисами)
7. [Конфигурация](#конфигурация)
8. [Развертывание](#развертывание)
9. [Мониторинг и логирование](#мониторинг-и-логирование)

## Обзор системы

Project Service - микросервис для управления проектами в системе управления проектами. Обеспечивает создание и управление проектами с поддержкой разграничения доступа на уровне владельцев.

### Ключевые возможности
- CRUD операции с проектами
- Поиск проектов по названию
- Контроль доступа на уровне владельца
- Интеграция с User Service
- Отслеживание времени создания/обновления
- Поддержка UUID для идентификаторов

## Архитектура

### Компоненты системы
1. **API Layer** (`src/api/routes.py`)
   - REST API endpoints
   - Валидация входных данных
   - Обработка ошибок
   - Маршрутизация

2. **Service Layer** (`src/auth.py`)
   - Интеграция с User Service
   - JWT аутентификация
   - Проверка токенов

3. **Data Layer** (`src/database.py`)
   - SQLAlchemy ORM
   - CRUD операции
   - Управление подключением к БД

4. **Models Layer**
   - API Models (`src/models/api_models.py`)
     - ProjectBase
     - ProjectCreate
     - ProjectUpdate
     - ProjectResponse
   - DB Models (`src/models/database_models.py`)
     - Project

### Структура проекта
```
project_service/
├── .env                       # Конфигурационные переменные
├── run.py                     # Точка входа приложения
├── docs/                      # Документация
└── src/
    ├── api/               
    │   └── routes.py          # API endpoints
    ├── models/
    │   ├── api_models.py      # Pydantic models
    │   └── database_models.py # SQLAlchemy models
    ├── auth.py                # Аутентификация
    ├── database.py            # Работа с БД
    └── main.py                # Конфигурация FastAPI
```

## Модели данных

### Project (Database Model)
```python
class Project(Base):
    id: UUID
    name: str           # max_length=100
    description: str    # Text field
    owner_id: UUID     
    created_at: datetime
    updated_at: datetime
```

### API Models
- ProjectBase: Базовые поля проекта
- ProjectCreate: Модель создания проекта
- ProjectUpdate: Модель обновления проекта
- ProjectResponse: Модель ответа API

## Аутентификация и безопасность

### JWT Authentication
- Интеграция с User Service
- Валидация JWT токенов
- Проверка активности пользователя

### Безопасность
- CORS middleware
- Валидация входных данных (Pydantic)
- Проверки прав доступа
- Защита от SQL-инъекций (через ORM)

## Взаимодействие с другими сервисами

### User Service
- Проверка аутентификации
- Получение информации о пользователе
- Проверка активности пользователя

## Конфигурация

### Переменные окружения (.env)
```
SECRET_KEY=your-secret-key
DATABASE_URL=postgresql://user:password@localhost/dbname
USER_SERVICE_URL=http://user-service:8000
```

## Развертывание

### Подготовка
1. Настройка окружения:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Конфигурация .env:
```
SECRET_KEY=your-secret-key
DATABASE_URL=postgresql://user:password@localhost/dbname
USER_SERVICE_URL=http://user-service:8000
```

### Запуск
```bash
python run.py
```

Сервис запустится на http://localhost:8001

## Мониторинг и логирование

### Логирование
- Стандартный вывод для операций
- Логирование ошибок БД
- Логирование проблем аутентификации
- Отслеживание времени выполнения

### Мониторинг
- Статус сервиса: GET /health
- Метрики БД
- Отслеживание ошибок

### Обработка ошибок
- HTTP ошибки (401, 403, 404, 500)
- Ошибки БД
- Ошибки аутентификации
- Недоступность User Service
