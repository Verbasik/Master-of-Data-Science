## Описание API

### Endpoints

#### POST /projects
Создание нового проекта
```json
Request:
{
    "name": "string",
    "description": "string"
}

Response:
{
    "id": "uuid",
    "name": "string",
    "description": "string",
    "created_at": "datetime",
    "updated_at": "datetime"
}
```

#### GET /projects
Получение списка проектов текущего пользователя
```json
Response:
[
    {
        "id": "uuid",
        "name": "string",
        "description": "string",
        "created_at": "datetime",
        "updated_at": "datetime"
    }
]
```

#### GET /projects/{project_id}
Получение информации о конкретном проекте
```json
Response:
{
    "id": "uuid",
    "name": "string",
    "description": "string",
    "created_at": "datetime",
    "updated_at": "datetime"
}
```

#### PUT /projects/{project_id}
Обновление проекта
```json
Request:
{
    "name": "string",
    "description": "string"
}

Response:
{
    "id": "uuid",
    "name": "string",
    "description": "string",
    "created_at": "datetime",
    "updated_at": "datetime"
}
```

#### DELETE /projects/{project_id}
Удаление проекта
```
Response: 204 No Content
```

#### GET /projects/search
Поиск проектов по названию
```
Query Parameters:
- name: string (поисковый запрос)

Response:
[
    {
        "id": "uuid",
        "name": "string",
        "description": "string",
        "created_at": "datetime",
        "updated_at": "datetime"
    }
]
```
