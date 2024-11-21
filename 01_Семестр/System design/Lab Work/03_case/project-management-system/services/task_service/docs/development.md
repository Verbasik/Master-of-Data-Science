# Task Service Documentation

## Оглавление
1. [Обзор системы](#обзор-системы)
2. [Архитектура сервиса](#архитектура-сервиса)
3. [Структура проекта](#структура-проекта)
4. [API Reference](#api-reference)
5. [Модели данных](#модели-данных)
6. [Аутентификация и безопасность](#аутентификация-и-безопасность)
7. [Взаимодействие с другими сервисами](#взаимодействие-с-другими-сервисами)
8. [Конфигурация](#конфигурация)
9. [Развертывание](#развертывание)

## Обзор системы
Task Service - микросервис для управления задачами в системе управления проектами. Обеспечивает полный жизненный цикл задач, включая создание, обновление, получение информации и удаление задач, с поддержкой статусов и приоритетов.

### Ключевые возможности
- Управление жизненным циклом задач
- Приоритизация задач
- Назначение исполнителей
- Фильтрация задач по проекту/исполнителю
- Интеграция с User Service для аутентификации
- Отслеживание времени создания/обновления
- Поддержка UUID для идентификаторов

### Статусы задач
- CREATED - Задача создана
- IN_PROGRESS - В работе
- ON_REVIEW - На проверке
- COMPLETED - Завершена

### Приоритеты задач
- LOW - Низкий
- MEDIUM - Средний (по умолчанию)
- HIGH - Высокий

## Архитектура сервиса

### Компоненты системы
1. **API Layer** (`src/api/routes.py`)
   - REST API endpoints
   - Валидация входных данных
   - Обработка HTTP запросов
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
   - DB Models (`src/models/database_models.py`)

## Структура проекта
```
task_service/
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

### Task (Database Model)
```python
class Task(Base):
    id: UUID
    title: str           # max_length=200
    description: str     # Text field
    status: TaskStatus
    priority: TaskPriority
    project_id: UUID
    creator_id: UUID
    assignee_id: Optional[UUID]
    created_at: datetime
    updated_at: datetime
```

### API Models
- TaskBase: Базовые поля задачи
- TaskCreate: Модель создания задачи
- TaskUpdate: Модель обновления задачи
- TaskResponse: Модель ответа API

## Аутентификация и безопасность

### JWT Authentication
- Интеграция с User Service
- Валидация JWT токенов
- Настраиваемое время жизни токенов

### Безопасность
- CORS middleware
- Валидация входных данных (Pydantic)
- Типизация данных
- Обработка ошибок

## Взаимодействие с другими сервисами

### User Service
- Аутентификация пользователей
- Получение информации о пользователях
- Проверка токенов

## Конфигурация

### Переменные окружения (.env)
```
SECRET_KEY=your-secret-key
DATABASE_URL=postgresql://user:password@localhost/dbname
USER_SERVICE_URL=http://user-service:8000
```

## Мониторинг и логирование

### Логирование
- Стандартный вывод для основных операций
- Логирование ошибок БД
- Логирование проблем аутентификации
- Отслеживание времени выполнения запросов

### Мониторинг
- Статус сервиса: GET /health
- Метрики БД
- Отслеживание времени ответа
- Мониторинг ошибок

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

Сервис запустится на http://localhost:8002
