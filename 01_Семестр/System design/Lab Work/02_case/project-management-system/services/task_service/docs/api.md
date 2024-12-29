## Описание API

### Endpoints

#### POST /tasks
Создание новой задачи
```json
Request:
{
    "title": "string",
    "description": "string",
    "priority": "low|medium|high",
    "project_id": "uuid",
    "assignee_id": "uuid"
}

Response:
{
    "id": "uuid",
    "title": "string",
    "description": "string",
    "priority": "string",
    "status": "string",
    "created_at": "datetime",
    "updated_at": "datetime",
    "project_id": "uuid",
    "assignee_id": "uuid",
    "creator_id": "uuid"
}
```

#### GET /tasks
Получение списка задач с возможностью фильтрации
```
Query Parameters:
- project_id: uuid (опционально)
- assignee_id: uuid (опционально)

Response:
[
    {
        "id": "uuid",
        "title": "string",
        "description": "string",
        "priority": "string",
        "status": "string",
        "created_at": "datetime",
        "updated_at": "datetime",
        "project_id": "uuid",
        "assignee_id": "uuid",
        "creator_id": "uuid"
    }
]
```

#### GET /tasks/{task_id}
Получение информации о конкретной задаче
```json
Response:
{
    "id": "uuid",
    "title": "string",
    "description": "string",
    "priority": "string",
    "status": "string",
    "created_at": "datetime",
    "updated_at": "datetime",
    "project_id": "uuid",
    "assignee_id": "uuid",
    "creator_id": "uuid"
}
```

#### PUT /tasks/{task_id}
Обновление задачи
```json
Request:
{
    "title": "string",
    "description": "string",
    "status": "string",
    "priority": "string",
    "assignee_id": "uuid"
}

Response:
{
    "id": "uuid",
    "title": "string",
    "description": "string",
    "priority": "string",
    "status": "string",
    "created_at": "datetime",
    "updated_at": "datetime",
    "project_id": "uuid",
    "assignee_id": "uuid",
    "creator_id": "uuid"
}
```

#### DELETE /tasks/{task_id}
Удаление задачи
```
Response: 204 No Content
```
