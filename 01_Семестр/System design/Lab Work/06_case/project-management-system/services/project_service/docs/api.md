## API Reference

### POST /projects
Создание нового проекта
```json
Request:
{
    "name": "string",         // 1-100 символов
    "description": "string"   // до 1000 символов
}

Response:
{
    "id": "uuid",
    "name": "string",
    "description": "string",
    "owner_id": "uuid",
    "created_at": "datetime",
    "updated_at": "datetime",
    "task_count": "number"    // опционально
}
```

### GET /projects
Получение списка проектов пользователя
```json
Response: Array[Project]
```

### GET /projects/{project_id}
Получение проекта по ID
```json
Response: Project
```

### PUT /projects/{project_id}
Обновление проекта
```json
Request:
{
    "name": "string",         // опционально
    "description": "string"   // опционально
}

Response: Project
```

### DELETE /projects/{project_id}
Удаление проекта
```
Response: 204 No Content
```

### GET /projects/search
Поиск проектов
```
Query Parameters:
- name: string (поисковый запрос)

Response: Array[Project]
```

## Примеры использования API

### Создание проекта
```python
import requests

# Создание проекта
response = requests.post(
    "http://localhost:8001/projects",
    headers={"Authorization": f"Bearer {token}"},
    json={
        "name": "New Project",
        "description": "Project description"
    }
)
project = response.json()
```

### Поиск проектов
```python
# Поиск проектов по имени
response = requests.get(
    "http://localhost:8001/projects/search",
    headers={"Authorization": f"Bearer {token}"},
    params={"name": "Project"}
)
projects = response.json()
```

### Обновление проекта
```python
# Обновление проекта
response = requests.put(
    f"http://localhost:8001/projects/{project_id}",
    headers={"Authorization": f"Bearer {token}"},
    json={
        "name": "Updated Project",
        "description": "New description"
    }
)
updated_project = response.json()
```