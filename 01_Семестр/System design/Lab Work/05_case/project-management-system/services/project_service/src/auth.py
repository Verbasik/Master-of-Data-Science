# project-management-system/services/project_service/src/auth.py
import os
import httpx
from dotenv import load_dotenv
from datetime import datetime, timedelta
from typing import Optional, Any
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from pydantic import BaseModel
from uuid import UUID

load_dotenv()

# Настройки JWT (должны быть идентичны с user_service)
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
USER_SERVICE_URL = os.getenv("USER_SERVICE_URL")

# Схема OAuth2 для получения токена
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class TokenData(BaseModel):
    """
    Description:
        Модель данных токена.

    Args:
        username (Optional[str]): Имя пользователя.

    Examples:
        >>> token_data = TokenData(username="john_doe")
        >>> token_data.username
        'john_doe'
    """
    username: Optional[str] = None

class User(BaseModel):
    """
    Description:
        Модель пользователя для аутентификации.

    Args:
        id (UUID): Уникальный идентификатор пользователя.
        username (str): Имя пользователя.
        email (str): Email пользователя.
        is_active (bool): Статус активности пользователя.

    Examples:
        >>> user = User(id=UUID('...'), username="john_doe", email="john@example.com", is_active=True)
        >>> user.username
        'john_doe'
    """
    id: UUID
    username: str
    email: str
    is_active: bool

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Description:
        Создание JWT токена.

    Args:
        data (dict): Данные для включения в токен.
        expires_delta (Optional[timedelta]): Время жизни токена.

    Returns:
        str: Закодированный JWT токен.

    Raises:
        ValueError: Если секретный ключ отсутствует.

    Examples:
        >>> token = create_access_token({"sub": "john_doe"})
        >>> len(token) > 0
        True
    """
    if not SECRET_KEY:
        raise ValueError("SECRET_KEY не установлен.")
        
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """
    Description:
        Получение текущего пользователя на основе JWT токена.

    Args:
        token (str): JWT токен.

    Returns:
        User: Объект пользователя.

    Raises:
        HTTPException: Если токен недействителен или пользователь не найден.

    Examples:
        >>> user = await get_current_user("valid_token")
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
            
        try:
            async with httpx.AsyncClient() as client:
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
                detail=f"Request error: {str(e)}"
            )
                
    except JWTError:
        raise credentials_exception
    except Exception as e:
        raise credentials_exception

async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """
    Description:
        Получение текущего активного пользователя.

    Args:
        current_user (User): Текущий пользователь.

    Returns:
        User: Объект активного пользователя.

    Raises:
        HTTPException: Если пользователь неактивен.

    Examples:
        >>> user = await get_current_active_user(User(...))
        >>> user.is_active
        True
    """
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user
