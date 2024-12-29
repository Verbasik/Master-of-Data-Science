# Project Management System

## Оглавление
1. [Описание проекта](#описание-проекта)
2. [Архитектура](#архитектура)
3. [Сервисы](#сервисы)
4. [Начало работы](#начало-работы)
5. [API Documentation](#api-documentation)
6. [Разработка](#разработка)
7. [Тестирование](#тестирование)

## Описание проекта
Project Management System - это микросервисная система управления проектами, состоящая из трех основных сервисов:
- User Service - управление пользователями и аутентификацией
- Project Service - управление проектами
- Task Service - управление задачами

Система позволяет создавать проекты, управлять задачами внутри проектов и контролировать доступ пользователей.

## Архитектура

### Структура проекта
```
project-management-system/
├── user_service/
│   ├── docs/
│   │   ├── api.md
│   │   └── development.md
│   └── src/
│       ├── api/
│       ├── auth.py
│       ├── database.py
│       ├── main.py
│       └── models.py
├── project_service/
│   ├── docs/
│   │   ├── api.md
│   │   └── development.md
│   └── src/
│       ├── api/
│       ├── auth.py
│       ├── database.py
│       ├── main.py
│       └── models.py
├── task_service/
│   ├── docs/
│   │   ├── api.md
│   │   └── development.md
│   └── src/
│       ├── api/
│       ├── auth.py
│       ├── database.py
│       ├── main.py
│       └── models.py
├── test_services.sh
└── README.md
```

### Технологический стек
- Python 3.7+
- FastAPI
- Pydantic
- JWT для аутентификации
- Uvicorn (ASGI сервер)

## Сервисы

### User Service (порт: 8000)
- Управление пользователями
- Аутентификация и авторизация
- Генерация JWT токенов
- [Подробная документация](./user_service/docs/api.md)

### Project Service (порт: 8001)
- Управление проектами
- Контроль доступа к проектам
- Поиск и фильтрация проектов
- [Подробная документация](./project_service/docs/api.md)

### Task Service (порт: 8002)
- Управление задачами
- Приоритизация и статусы задач
- Назначение исполнителей
- [Подробная документация](./task_service/docs/api.md)

## Начало работы

### Предварительные требования
```bash
# Установка зависимостей
pip install -r requirements.txt
```

### Запуск сервисов
```bash
# Запуск User Service
cd user_service/src
uvicorn main:app --host 0.0.0.0 --port 8000

# Запуск Project Service
cd project_service/src
uvicorn main:app --host 0.0.0.0 --port 8001

# Запуск Task Service
cd task_service/src
uvicorn main:app --host 0.0.0.0 --port 8002
```

## API Documentation

После запуска сервисов, документация Swagger UI доступна по следующим адресам:
- User Service: http://localhost:8000/docs
- Project Service: http://localhost:8001/docs
- Task Service: http://localhost:8002/docs

## Разработка

### Структура каждого сервиса
- `main.py` - точка входа приложения
- `models.py` - модели данных
- `auth.py` - аутентификация и авторизация
- `database.py` - работа с базой данных
- `api/routes.py` - маршруты API

### Конвенции кода
- Используйте типизацию Python
- Следуйте PEP 8
- Документируйте все публичные функции и классы
- Используйте async/await где это возможно

## Тестирование

### Интеграционное тестирование
```bash
# Запуск тестового сценария
./test_services.sh
```

Тестовый сценарий проверяет:
1. Создание пользователя
2. Получение токена аутентификации
3. Создание проекта
4. Создание задачи
5. Проверку созданных данных
6. Получение списков проектов и задач

### Результаты тестирования
- Результаты сохраняются в `test_results.log`
- Успешные операции отмечаются зеленым цветом
- Ошибки отмечаются красным цветом
