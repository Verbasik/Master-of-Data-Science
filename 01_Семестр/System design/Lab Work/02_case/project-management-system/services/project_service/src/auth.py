# project-management-system/services/project_service/src/auth.py
from datetime import datetime, timedelta
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from pydantic import BaseModel
from uuid import UUID

# Настройки JWT - должны быть идентичны с user_service
SECRET_KEY = "your-secret-key"  # В реальном приложении должен храниться в переменных окружения
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Схема OAuth2 для получения токена
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class TokenData(BaseModel):
    """
    Описание:
      Модель данных токена.

    Атрибуты:
        username (Optional[str]): Имя пользователя.

    Пример использования:
        >>> token_data = TokenData(username="john_doe")
        >>> token_data.username
        'john_doe'
    """
    username: Optional[str] = None

class User(BaseModel):
    """
    Описание:
      Модель пользователя для аутентификации.

    Атрибуты:
        id (UUID): Уникальный идентификатор пользователя.
        username (str): Имя пользователя.
        email (str): Email пользователя.
        is_active (bool): Статус активности пользователя.

    Пример использования:
        >>> user = User(id=UUID('...'), username="john_doe", email="john@example.com", is_active=True)
        >>> user.username
        'john_doe'
    """
    id: UUID
    username: str
    email: str
    is_active: bool

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """
    Описание:
      Создание JWT токена.

    Аргументы:
        data (dict): Данные для включения в токен.
        expires_delta (Optional[timedelta]): Время жизни токена.

    Возвращает:
        str: Закодированный JWT токен.

    Пример использования:
        >>> token = create_access_token({"sub": "john_doe"})
        >>> len(token) > 0
        True
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """
    Описание:
      Получение текущего пользователя на основе JWT токена.

    Аргументы:
        token (str): JWT токен.

    Возвращает:
        User: Объект пользователя.

    Исключения:
        HTTPException: Если токен недействителен или пользователь не найден.

    Пример использования:
        >>> user = await get_current_user(valid_token)
        >>> user.username
        'john_doe'
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
    
    # Здесь должен быть запрос к user_service для получения информации о пользователе
    # В данном примере мы создаем фиктивного пользователя
    user = User(
        id=UUID('12345678-1234-5678-1234-567812345678'),
        username=token_data.username,
        email=f"{token_data.username}@example.com",
        is_active=True
    )
    
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """
    Описание:
      Получение текущего активного пользователя.

    Аргументы:
        current_user (User): Текущий пользователь.

    Возвращает:
        User: Объект активного пользователя.

    Исключения:
        HTTPException: Если пользователь неактивен.

    Пример использования:
        >>> user = await get_current_active_user(current_user)
        >>> user.is_active
        True
    """
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user