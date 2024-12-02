# User Service Documentation

## Оглавление
1. [Обзор системы](#обзор-системы)
2. [Архитектура сервиса](#архитектура-сервиса)
3. [Структура проекта](#структура-проекта)
4. [API Reference](#api-reference)
5. [Модели данных](#модели-данных)
6. [Аутентификация и безопасность](#аутентификация-и-безопасность)
7. [Кеширование](#кеширование)
8. [Конфигурация](#конфигурация)
9. [Используемые технологии](#используемые-технологии)

## Обзор системы
User Service представляет собой микросервис для управления пользователями в рамках системы управления проектами. Сервис обеспечивает функциональность управления пользователями, включая регистрацию, аутентификацию и авторизацию с использованием JWT токенов, а также оптимизацию производительности через Redis кеширование.

### Ключевые возможности
- Регистрация и управление пользователями
- JWT-based аутентификация и авторизация
- Кеширование данных пользователей в Redis
- Реализация паттерна "сквозное чтение" (Read-Through)
- Административное управление пользователями
- Автоматическое создание admin-пользователя при первом запуске
- Отслеживание времени создания и обновления записей
- Поддержка UUID для идентификаторов

## Архитектура сервиса

### Компоненты системы
1. **API Layer** (`src/api/routes.py`)
   - REST API endpoints
   - Валидация входных данных
   - Маршрутизация запросов

2. **Service Layer** (`src/auth.py`)
   - Управление JWT токенами
   - Хеширование паролей (bcrypt)
   - Аутентификация пользователей

3. **Cache Layer** (`src/cache.py`)
   - Управление Redis кешем
   - Реализация паттерна "сквозное чтение"
   - Сериализация/десериализация данных
   
4. **Data Layer** (`src/database.py`)
   - Интеграция с SQLAlchemy
   - CRUD операции
   - Управление подключением к БД

5. **Models Layer**
   - API Models (`src/models/api_models.py`) - Pydantic models
   - DB Models (`src/models/database_models.py`) - PostgreSQL models

## Структура проекта
```
user_service/
├── .env                       # Конфигурационные переменные
├── run.py                     # Точка входа приложения
├── docs/                      # Документация
└── src/
    ├── api/               
    │   └── routes.py          # API endpoints
    ├── models/
    │   ├── api_models.py      # Pydantic models
    │   └── database_models.py # PostgreSQL models
    ├── auth.py                # Аутентификация
    ├── cache.py               # Работа с Redis кешем
    ├── database.py            # Работа с БД
    └── main.py                # Конфигурация FastAPI
```

## Кеширование

### Паттерн "Сквозное чтение" (Read-Through)
Сервис реализует паттерн "сквозное чтение" для оптимизации доступа к данным пользователей:

1. При запросе данных пользователя:
   - Сначала проверяется наличие данных в Redis кеше
   - Если данные найдены, они возвращаются из кеша
   - Если данных нет, они загружаются из PostgreSQL
   - Загруженные данные сохраняются в кеш и возвращаются клиенту

2. При создании пользователя:
   - Данные сохраняются в PostgreSQL
   - Затем автоматически кешируются в Redis

3. При обновлении пользователя:
   - Данные обновляются в PostgreSQL
   - Кеш инвалидируется для обеспечения консистентности

### Конфигурация Redis
- Настраиваемое время жизни кеша (TTL)
- Автоматическая сериализация/десериализация данных
- Поддержка отказоустойчивости

## Модели данных

### UserBase (Pydantic)
```python
class UserBase(BaseModel):
    username: str
    email: EmailStr
```

### User (Database)
```python
class User(Base):
    id: UUID
    username: str
    email: str
    hashed_password: str
    is_active: bool = True
    is_admin: bool = False
    created_at: datetime
    updated_at: datetime
```

## Аутентификация и безопасность

### JWT Authentication
- Использование OAuth2 с JWT токенами
- Настраиваемое время жизни токенов (ACCESS_TOKEN_EXPIRE_MINUTES)
- Хеширование паролей с использованием bcrypt
- Проверка срока действия токенов

### Безопасность
- CORS middleware с настраиваемыми правилами
- Валидация входных данных через Pydantic
- Безопасное хранение паролей (bcrypt)
- Отслеживание времени создания и обновления записей

## Конфигурация

### Переменные окружения (.env)
```
SECRET_KEY=your-secret-key
ACCESS_TOKEN_EXPIRE_MINUTES=30
DATABASE_URL=postgresql://user:password@localhost/dbname

# Redis кеширование
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_PASSWORD=your-redis-password
REDIS_TTL=3600
```

## Развертывание

### Подготовка к запуску
1. Создайте виртуальное окружение:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

2. Установите зависимости:
```bash
pip install -r requirements.txt
```

3. Настройте переменные окружения в .env файле:
```
SECRET_KEY=your-secret-key
ACCESS_TOKEN_EXPIRE_MINUTES=30
DATABASE_URL=postgresql://user:password@localhost/dbname

# Redis кеширование
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_PASSWORD=your-redis-password
REDIS_TTL=3600
```

### Запуск сервиса
```bash
python run.py
```

Сервис будет доступен по адресу: http://localhost:8000

## Используемые технологии

### Core
- Python 3.12+
- FastAPI
- SQLAlchemy
- Pydantic
- Uvicorn
- Redis

### Security
- python-jose[cryptography]
- passlib[bcrypt]
- python-multipart

### Caching
- redis-py
- msgpack или json (для сериализации)

### Validation
- email-validator
- pydantic[email]

### Development
- python-dotenv
- uvicorn[standard]
