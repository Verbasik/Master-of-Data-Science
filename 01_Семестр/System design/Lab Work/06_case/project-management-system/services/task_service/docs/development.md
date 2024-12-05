# Task Service Documentation

---

#### 1. Обзор системы

**Task Service** - микросервис для управления задачами в системе управления проектами. Он обеспечивает полный жизненный цикл задач, включая создание, обновление, получение информации и удаление задач, с поддержкой статусов и приоритетов.

**Ключевые возможности:**
- Управление жизненным циклом задач
- Приоритизация задач
- Назначение исполнителей
- Фильтрация задач по проекту/исполнителю
- Интеграция с User Service для аутентификации
- Интеграция с Kafka для обработки событий
- Поддержка асинхронных операций через MongoDB
- Отслеживание времени создания/обновления
- Поддержка UUID для идентификаторов

**Статусы задач:**
- CREATED - Задача создана
- IN_PROGRESS - В работе
- ON_REVIEW - На проверке
- COMPLETED - Завершена

**Приоритеты задач:**
- LOW - Низкий
- MEDIUM - Средний (по умолчанию)
- HIGH - Высокий

---

#### 2. Архитектура сервиса

**Компоненты системы:**
1. **API Layer** (`src/api/routes.py`):
   - REST API endpoints
   - Валидация входных данных
   - Обработка HTTP запросов
   - Маршрутизация

2. **Service Layer** (`src/services/task_service.py`):
   - Бизнес-логика управления задачами
   - CRUD операции с задачами

3. **Data Layer** (`src/database.py`):
   - Подключение и управление MongoDB
   - Асинхронные операции с базой данных

4. **Models Layer**:
   - **API Models** (`src/models/api_models.py`): Модели для взаимодействия с API.
   - **Mongo Models** (`src/models/mongo_models.py`): Модели для работы с MongoDB.

5. **Authentication** (`src/auth.py`):
   - JWT аутентификация
   - Взаимодействие с User Service для проверки токенов

6. **Event Processing** (`src/events/kafka_producer.py`, `src/events/event_handler.py`):
   - Kafka Producer: отправка событий о создании задач.
   - Kafka Event Handler: обработка событий и запись данных в MongoDB.

---

#### 3. Структура проекта

```
task_service/
├── .env                       # Конфигурационные переменные
├── run.py                     # Точка входа приложения
├── src/
│   ├── api/
│   │   └── routes.py          # API endpoints
│   ├── models/
│   │   ├── api_models.py      # Pydantic models для API
│   │   └── mongo_models.py    # Pydantic models для MongoDB
│   ├── services/
│   │   └── task_service.py    # Бизнес-логика
│   ├── events/
│   │   ├── kafka_producer.py  # Kafka Producer
│   │   └── event_handler.py   # Kafka Event Handler
│   ├── auth.py                # Аутентификация
│   ├── database.py            # Работа с MongoDB
│   └── main.py                # Конфигурация FastAPI
└── core/
    └── config.py              # Настройки приложения
```

---

#### 4. API Reference

**1. Создание задачи**
- **Endpoint**: `/tasks`
- **Method**: POST
- **Request Body**: `TaskCreate`
- **Response**: `TaskResponse`
- **Authentication**: Требуется JWT токен

**2. Получение списка задач**
- **Endpoint**: `/tasks`
- **Method**: GET
- **Query Parameters**: `project_id`, `assignee_id`, `status`, `priority`
- **Response**: `List[TaskResponse]`
- **Authentication**: Требуется JWT токен

**3. Получение задачи по идентификатору**
- **Endpoint**: `/tasks/{task_id}`
- **Method**: GET
- **Path Parameters**: `task_id`
- **Response**: `TaskResponse`
- **Authentication**: Требуется JWT токен

**4. Обновление задачи**
- **Endpoint**: `/tasks/{task_id}`
- **Method**: PUT
- **Path Parameters**: `task_id`
- **Request Body**: `TaskUpdate`
- **Response**: `TaskResponse`
- **Authentication**: Требуется JWT токен

**5. Удаление задачи**
- **Endpoint**: `/tasks/{task_id}`
- **Method**: DELETE
- **Path Parameters**: `task_id`
- **Response**: 204 No Content
- **Authentication**: Требуется JWT токен

**6. Обновление статуса задачи**
- **Endpoint**: `/tasks/{task_id}/status`
- **Method**: PATCH
- **Path Parameters**: `task_id`
- **Request Body**: `status`
- **Response**: `TaskResponse`
- **Authentication**: Требуется JWT токен

**7. Обновление исполнителя задачи**
- **Endpoint**: `/tasks/{task_id}/assignee`
- **Method**: PATCH
- **Path Parameters**: `task_id`
- **Request Body**: `assignee_id`
- **Response**: `TaskResponse`
- **Authentication**: Требуется JWT токен

---

#### 5. Модели данных

**1. TaskCreate**
```python
class TaskCreate(BaseModel):
    title: str
    description: str
    priority: TaskPriority
    project_id: UUID
    assignee_id: Optional[UUID] = None
```

**2. TaskUpdate**
```python
class TaskUpdate(BaseModel):
    title: Optional[str]
    description: Optional[str]
    status: Optional[TaskStatus]
    priority: Optional[TaskPriority]
    assignee_id: Optional[UUID]
```

**3. TaskResponse**
```python
class TaskResponse(BaseModel):
    id: UUID
    title: str
    description: str
    status: TaskStatus
    priority: TaskPriority
    project_id: UUID
    creator_id: UUID
    assignee_id: Optional[UUID]
    created_at: datetime
    updated_at: datetime
```

---

#### 6. Аутентификация и безопасность

**JWT Authentication:**
- Используется для аутентификации пользователей.
- Токены проверяются с помощью User Service.

**Безопасность:**
- Используется CORS middleware для управления разрешениями.
- Валидация входных данных с помощью Pydantic.
- Логирование ошибок и операций.

---

#### 7. Взаимодействие с другими сервисами

**User Service:**
- Аутентификация пользователей через JWT.
- Проверка токенов и получение информации о пользователе.

**Kafka:**
- Producer отправляет события task_created при создании задачи.
- Event Handler обрабатывает события и сохраняет их в MongoDB.

---

#### 8. Конфигурация

**Переменные окружения (.env):**
```
SECRET_KEY=your-secret-key
MONGODB_URL=mongodb://mongodb:27017
MONGODB_DB_NAME=task_service
USER_SERVICE_URL=http://user-service:8000
PROJECT_SERVICE_URL=http://project-service:8001
TASK_SERVICE_URL=http://task-service:8002
KAFKA_BOOTSTRAP_SERVERS = "kafka:29092"
KAFKA_TASK_TOPIC = "task_events"
KAFKA_CONSUMER_GROUP = "task_service_group"
KAFKA_MAX_RETRIES = 3
KAFKA_RETRY_DELAY = 1
```

---

#### 9. Развертывание

**Подготовка:**
1. Создайте виртуальное окружение:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
2. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```
3. Настройте переменные окружения в файле `.env`.

**Запуск:**
```bash
python run.py
```
Сервис будет доступен по адресу `http://localhost:8002`.

---

#### 10. Мониторинг и логирование

**Логирование:**
- Используются стандартные логи для отслеживания операций и ошибок.
- Логи выводятся в консоль и включают информацию о запросах, ошибках и операциях с базой данных.

**Мониторинг:**
- Здоровье сервиса можно проверить по эндпоинту `/health`, который возвращает `{"status": "healthy"}`.
- Отслеживается время выполнения запросов и ошибки базы данных.