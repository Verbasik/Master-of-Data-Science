# project-management-system/services/task_service/src/auth.py
from datetime import datetime, timedelta
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from pydantic import BaseModel
from uuid import UUID

# Настройки JWT (должны быть идентичны с user_service)
SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class TokenData(BaseModel):
    """
    Описание:
      Модель данных токена.

    Атрибуты:
        username: Имя пользователя
    """
    username: Optional[str] = None

class User(BaseModel):
    """
    Описание:
      Модель пользователя для аутентификации.

    Атрибуты:
        id: ID пользователя
        username: Имя пользователя
        email: Email пользователя
        is_active: Статус активности
    """
    id: UUID
    username: str
    email: str
    is_active: bool

async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """
    Описание:
      Получение текущего пользователя из токена.

    Args:
        token: JWT токен

    Returns:
        User: Объект пользователя

    Raises:
        HTTPException: При ошибке аутентификации
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception

    # В реальном приложении здесь должен быть запрос к user_service
    user = User(
        id=UUID('12345678-1234-5678-1234-567812345678'),
        username=token_data.username,
        email=f"{token_data.username}@example.com",
        is_active=True
    )
    
    if user is None:
        raise credentials_exception
    return user