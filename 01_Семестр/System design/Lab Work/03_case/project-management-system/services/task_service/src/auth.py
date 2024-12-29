# project-management-system/services/task_service/src/auth.py

# --- Импорты стандартных библиотек ---
import os
from datetime import datetime, timedelta

# --- Импорты сторонних библиотек ---
from dotenv import load_dotenv
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from pydantic import BaseModel
from uuid import UUID
import httpx

# Загрузка переменных окружения
load_dotenv()

# --- Настройки JWT ---
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
USER_SERVICE_URL = os.getenv("USER_SERVICE_URL")

# --- Настройка OAuth2 ---
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class TokenData(BaseModel):
    """
    Description:
        Модель данных токена.

    Args:
        username: Имя пользователя (опционально)
    """
    username: Optional[str] = None

class User(BaseModel):
    """
    Description:
        Модель пользователя для аутентификации.

    Args:
        id: ID пользователя
        username: Имя пользователя
        email: Email пользователя
        is_active: Статус активности пользователя
    """
    id: UUID
    username: str
    email: str
    is_active: bool

async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """
    Description:
        Получение текущего пользователя из токена.

    Args:
        token: JWT токен

    Returns:
        User: Объект пользователя

    Raises:
        HTTPException: Ошибки аутентификации или недоступность сервиса
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: Optional[str] = payload.get("sub")
        if username is None:
            raise credentials_exception

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{USER_SERVICE_URL}/users/me",
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=10.0
                )

                if response.status_code == 200:
                    user_data = response.json()
                    return User(**user_data)
                else:
                    raise credentials_exception
            except httpx.RequestError as e:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"User service unavailable: {str(e)}"
                )
    except JWTError:
        raise credentials_exception
