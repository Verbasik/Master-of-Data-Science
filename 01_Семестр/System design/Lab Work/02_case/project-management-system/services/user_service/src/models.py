# project-management-system/services/user_service/src/models.py
# ============================
# БЛОК ИМПОРТОВ
# ============================
# Импорты из сторонних библиотек
# Эти модули используются для валидации данных и работы с аннотациями типов.
from pydantic import BaseModel, EmailStr, Field  # Базовая модель и поля для валидации данных

# Импорты из стандартной библиотеки
# Используются для работы с UUID и опциональными типами.
from typing import Optional                     # Опциональные типы для аннотаций
from uuid import UUID, uuid4                    # Генерация и обработка UUID


class UserBase(BaseModel):
    """
    Description:
      Базовая модель пользователя.

    Args:
        username: Имя пользователя.
        email: Email пользователя.

    Examples:
        >>> user = UserBase(username="john_doe", email="john@example.com")
        >>> user.dict()
        {'username': 'john_doe', 'email': 'john@example.com'}
    """
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr

class UserCreate(UserBase):
    """
    Description:
      Модель для создания пользователя.

    Args:
        username: Имя пользователя.
        email: Email пользователя.
        password: Пароль пользователя.

    Examples:
        >>> user = UserCreate(username="john_doe", email="john@example.com", password="secretpass")
        >>> user.dict()
        {'username': 'john_doe', 'email': 'john@example.com', 'password': 'secretpass'}
    """
    password: str = Field(..., min_length=8)

class User(UserBase):
    """
    Description:
      Полная модель пользователя.

    Args:
        id: Уникальный идентификатор пользователя.
        username: Имя пользователя.
        email: Email пользователя.
        is_active: Статус активности пользователя.

    Examples:
        >>> user = User(id=uuid4(), username="john_doe", email="john@example.com", is_active=True)
        >>> user.dict()
        {'id': UUID('...'), 'username': 'john_doe', 'email': 'john@example.com', 'is_active': True}
    """
    id: UUID = Field(default_factory=uuid4)
    is_active: bool = True

class UserInDB(User):
    """
    Description:
      Модель пользователя для хранения в базе данных.

    Args:
        id: Уникальный идентификатор пользователя.
        username: Имя пользователя.
        email: Email пользователя.
        is_active: Статус активности пользователя.
        hashed_password: Хешированный пароль пользователя.

    Examples:
        >>> user = UserInDB(id=uuid4(), username="john_doe", email="john@example.com", is_active=True, hashed_password="hashedpass")
        >>> user.dict()
        {'id': UUID('...'), 'username': 'john_doe', 'email': 'john@example.com', 'is_active': True, 'hashed_password': 'hashedpass'}
    """
    hashed_password: str

class Token(BaseModel):
    """
    Description:
      Модель токена доступа.

    Args:
        access_token: Токен доступа.
        token_type: Тип токена.

    Examples:
        >>> token = Token(access_token="eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...", token_type="bearer")
        >>> token.dict()
        {'access_token': 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...', 'token_type': 'bearer'}
    """
    access_token: str
    token_type: str