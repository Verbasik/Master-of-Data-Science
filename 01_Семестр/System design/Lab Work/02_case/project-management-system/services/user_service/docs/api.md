## Описание API

### Endpoints

#### POST /users
Создание нового пользователя
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
    "is_active": "boolean"
}
```

#### GET /users
Получение списка пользователей
```json
Response:
[
    {
        "id": "uuid",
        "username": "string",
        "email": "string",
        "is_active": "boolean"
    }
]
```

#### GET /users/{user_id}
Получение информации о конкретном пользователе
```json
Response:
{
    "id": "uuid",
    "username": "string",
    "email": "string",
    "is_active": "boolean"
}
```

#### POST /token
Получение токена доступа
```json
Request:
{
    "username": "string",
    "password": "string"
}

Response:
{
    "access_token": "string",
    "token_type": "string"
}
```

#### GET /users/me
Получение информации о текущем пользователе
```json
Response:
{
    "id": "uuid",
    "username": "string",
    "email": "string",
    "is_active": "boolean"
}
```