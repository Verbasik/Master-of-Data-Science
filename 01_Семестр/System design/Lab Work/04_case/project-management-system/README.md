# Project Management System

## Оглавление
1. [Обзор системы](#обзор-системы)
2. [Архитектура](#архитектура)
3. [Микросервисы](#микросервисы)
4. [Развертывание](#развертывание)
5. [Тестирование](#тестирование)
6. [Безопасность](#безопасность)
7. [Мониторинг](#мониторинг)
8. [Разработка](#разработка)

## Обзор системы

Project Management System - это микросервисная система управления проектами, состоящая из трех основных сервисов и базы данных PostgreSQL. Система обеспечивает полный цикл управления проектами и задачами с поддержкой многопользовательского режима.

### Ключевые возможности
- Управление пользователями и аутентификацией
- Создание и управление проектами
- Управление задачами с приоритетами и статусами
- REST API для всех операций
- Docker-based развертывание
- Автоматизированное тестирование

## Архитектура

### Компоненты системы
1. **User Service** (Порт 8000)
   - Управление пользователями
   - JWT аутентификация
   - Авторизация

2. **Project Service** (Порт 8001)
   - Управление проектами
   - Права доступа
   - Интеграция с User Service

3. **Task Service** (Порт 8002)
   - Управление задачами
   - Статусы: CREATED, IN_PROGRESS, ON_REVIEW, COMPLETED
   - Приоритеты: LOW, MEDIUM, HIGH
   - Интеграция с MongoDB
   - JWT аутентификация

4. **PostgreSQL** (Порт 5432)
   - Хранение данных всех сервисов
   - Миграции через Alembic
   - Общая схема данных

5. **MongoDB** (Порт 27017)
   - Хранение данных Task Service
   - Асинхронные операции
   - Поддержка UUID

### Структура проекта
```
project-management-system/
├── docker-compose.yml           # Docker конфигурация
├── init-scripts/                # Скрипты инициализации БД
├── services/
│   ├── user_service/
│   ├── project_service/
│   ├── task_service/
│   └── Dockerfile              # Multi-stage сборка
├── test_services.sh            # Скрипт тестирования
└── README.md                   # Документация
```

## Развертывание

### Предварительные требования
- Docker и Docker Compose
- PostgreSQL 13+
- MongoDB 4.4+
- Python 3.12+
- Bash (для тестирования)

### Установка и запуск

1. Клонирование репозитория:
```bash
git clone <repository-url>
cd project-management-system
```

2. Настройка переменных окружения:
```bash
# .env файл в корне проекта
# Database Settings
POSTGRES_USER=admin
POSTGRES_PASSWORD=secret
POSTGRES_DB=project_management
DATABASE_URL=postgresql://admin:secret@localhost:5432/project_management

# MongoDB Settings
MONGO_USER=mongoadmin
MONGO_PASSWORD=mongopass
MONGO_DB=task_service
MONGODB_URL=mongodb://mongoadmin:mongopass@mongodb:27017/task_service?authSource=admin
MONGODB_DB_NAME=task_service

# Application Settings
SECRET_KEY=your-secret-key-here
DEBUG=false
ENVIRONMENT=production
BUILD_ENV=production

# Service URLs (internal)
USER_SERVICE_URL=http://localhost:8000
PROJECT_SERVICE_URL=http://localhost:8001
TASK_SERVICE_URL=http://localhost:8002
```

3. Запуск через Docker Compose:
```bash
docker compose up --build
```

### Проверка работоспособности
```bash
# Проверка статуса сервисов
docker compose ps

# Проверка логов
docker compose logs -f
```

## Тестирование

### Автоматизированное тестирование
Система включает скрипт комплексного тестирования `test_services.sh`, который проверяет:

### 1. Проверка окружения
- Валидация .env файла
- Проверка обязательных переменных
- Настройка подключений к БД

### 2. Проверка сервисов
- User Service (порт 8000)
- Project Service (порт 8001)
- Task Service (порт 8002)
- Эндпоинт /health для каждого сервиса

### 3. Проверка PostgreSQL
- Подключение к БД
- Конфигурация подключения
- Валидация структуры данных

### 4. Проверка MongoDB
- Подключение к БД
- Валидация индексов:
  - Одиночные: project_id, creator_id, assignee_id, status, priority, created_at
  - Составной: {project_id: 1, status: 1, priority: -1}

### 5. Функциональное тестирование
- Создание пользователя
- JWT аутентификация
- Создание проекта
- Создание и верификация задачи
- Проверка сохранения в MongoDB

```bash
# Запуск тестов
./test_services.sh

# Результаты
cat test_results.log
```

## Безопасность

### Аутентификация
- JWT токены для всех сервисов
- Хеширование паролей (bcrypt)
- OAuth2 с Bearer токенами

### Права доступа
- Разграничение на уровне пользователей
- Проверка владельца проекта
- Валидация всех запросов

### Сетевая безопасность
- CORS настройки
- HTTPS поддержка
- Rate limiting

### Мониторинг проблем
1. Проверка статуса сервисов
```bash
curl http://localhost:8000/health
curl http://localhost:8001/health
curl http://localhost:8002/health
```

2. Проверка логов
```bash
# Последние ошибки
docker compose logs --tail=100 service-name | grep ERROR

# Логи конкретного контейнера
docker logs container-id
```

3. Метрики системы
```bash
# Использование ресурсов
docker stats

# Статистика БД
psql -c "SELECT * FROM pg_stat_activity;"
```

## Разработка

### Требования к окружению
- Python 3.12+
- Poetry или pip
- PostgreSQL 13+
- MongoDB 4.4+
- Docker и Docker Compose

### Рекомендации по разработке
- Следуйте PEP 8
- Используйте типизацию
- Документируйте API
- Пишите тесты

### Процесс разработки
1. Создайте ветку для фичи
2. Разработайте функционал
3. Добавьте тесты
4. Создайте PR
5. После проверки сделайте merge
