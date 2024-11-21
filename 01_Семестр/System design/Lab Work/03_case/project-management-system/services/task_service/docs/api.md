## API Reference

### POST /tasks
Создание новой задачи
```json
Request:
{
    "title": "string",         // 1-200 символов
    "description": "string",   // до 2000 символов
    "priority": "low|medium|high",
    "project_id": "uuid",
    "assignee_id": "uuid"      // опционально
}

Response:
{
    "id": "uuid",
    "title": "string",
    "description": "string",
    "priority": "string",
    "status": "created",
    "project_id": "uuid",
    "creator_id": "uuid",
    "assignee_id": "uuid",
    "created_at": "datetime",
    "updated_at": "datetime"
}
```

### GET /tasks
Получение списка задач
```
Query Parameters:
- project_id: uuid (опционально)
- assignee_id: uuid (опционально)

Response: Array[Task]
```

### GET /tasks/{task_id}
Получение задачи по ID
```json
Response: Task
```

### PUT /tasks/{task_id}
Обновление задачи
```json
Request:
{
    "title": "string",         // опционально
    "description": "string",   // опционально
    "status": "string",        // опционально
    "priority": "string",      // опционально
    "assignee_id": "uuid"      // опционально
}

Response: Task
```

### DELETE /tasks/{task_id}
Удаление задачи
```
Response: 204 No Content
```
