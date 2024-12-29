## API Reference

### Endpoints

#### POST /users
Создание нового пользователя (с автоматическим кешированием)
```json
Request:
{
    "username": "string",
    "email": "string",
    "password": "string"
}

Response:
{
    "id": "uuid",
    "username": "string",
    "email": "string",
    "is_active": "boolean",
    "created_at": "datetime",
    "updated_at": "datetime"
}
```

#### GET /users
Получение списка пользователей (требует аутентификации)
```json
Response:
[
    {
        "id": "uuid",
        "username": "string",
        "email": "string",
        "is_active": "boolean",
        "created_at": "datetime",
        "updated_at": "datetime"
    }
]
```

#### GET /users/me
Получение информации о текущем пользователе (требует аутентификации)
```json
Response:
{
    "id": "uuid",
    "username": "string",
    "email": "string",
    "is_active": "boolean",
    "created_at": "datetime",
    "updated_at": "datetime"
}
```

#### POST /token
Получение JWT токена
```json
Request:
{
    "username": "string",
    "password": "string"
}

Response:
{
    "access_token": "string",
    "token_type": "bearer"
}
```

#### GET /users/{user_id}
Получение пользователя (с использованием Read-Through кеширования)
```json
Response:
{
    "id": "uuid",
    "username": "string",
    "email": "string",
    "is_active": "boolean",
    "created_at": "datetime",
    "updated_at": "datetime"
}
```

#### PUT /users/{user_id}
Обновление пользователя (с инвалидацией кеша)
```json
Request:
{
    "username": "string",
    "email": "string",
    "is_active": "boolean"
}

Response:
{
    "id": "uuid",
    "username": "string",
    "email": "string",
    "is_active": "boolean",
    "created_at": "datetime",
    "updated_at": "datetime"
}
```