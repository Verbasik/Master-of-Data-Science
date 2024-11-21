# project-management-system/services/user_service/src/models/api_models.py
from pydantic import BaseModel, EmailStr, Field
from uuid import UUID, uuid4
from datetime import datetime

class UserBase(BaseModel):
    """
    Description:
        Базовая модель пользователя.

    Attributes:
        username (str): Уникальное имя пользователя.
        email (EmailStr): Электронная почта пользователя.
    """
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr

    class Config:
        from_attributes = True


class UserCreate(UserBase):
    """
    Description:
        Модель для создания нового пользователя.

    Attributes:
        password (str): Пароль пользователя (минимум 8 символов).
    """
    password: str = Field(..., min_length=8)


class User(UserBase):
    """
    Description:
        Модель пользователя с дополнительными атрибутами.

    Attributes:
        id (UUID): Уникальный идентификатор пользователя.
        is_active (bool): Статус активности пользователя.
        created_at (datetime): Время создания записи о пользователе.
        updated_at (datetime): Время последнего обновления записи о пользователе.
    """
    id: UUID = Field(default_factory=uuid4)
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        """Настройки для совместимости с ORM."""
        orm_mode = True


class UserInDB(User):
    """
    Description:
        Модель пользователя в базе данных с хэшированным паролем.

    Attributes:
        hashed_password (str): Хэшированный пароль пользователя.
    """
    hashed_password: str


class Token(BaseModel):
    """
    Description:
        Модель для токена аутентификации.

    Attributes:
        access_token (str): Токен доступа для аутентификации.
        token_type (str): Тип токена (по умолчанию "bearer").
    """
    access_token: str
    token_type: str = "bearer"
