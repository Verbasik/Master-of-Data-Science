# project-management-system/services/user_service/src/api/routes.py

# ============================
# БЛОК ИМПОРТОВ
# ============================

# --- Импорты сторонних библиотек ---
from uuid import UUID
from typing import List
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

# --- Импорты внутренних модулей проекта ---
from ..models.api_models import User, UserCreate, Token
from ..models.database_models import User as UserDB
from ..database import db
from ..auth import (
    authenticate_user, create_access_token,
    get_current_user, get_password_hash,
    ACCESS_TOKEN_EXPIRE_MINUTES
)

# Инициализация маршрутизатора
router = APIRouter()

@router.post("/users", response_model=User, status_code=status.HTTP_201_CREATED)
async def create_user(user: UserCreate):
    """
    Description:
        Создание нового пользователя.

    Args:
        user: Данные для создания пользователя.

    Returns:
        Созданный объект пользователя.

    Raises:
        HTTPException: Если пользователь с таким именем уже существует.

    Examples:
        >>> response = client.post("/users", json={"username": "newuser", "email": "new@example.com", "password": "newpass"})
        >>> response.status_code
        201
        >>> response.json()["username"]
        'newuser'
    """
    db_user = db.get_user_by_username(UserDB, user.username)
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    user_data = {
        "username": user.username,
        "email": user.email,
        "hashed_password": get_password_hash(user.password),
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
    db_user = db.create(UserDB, user_data)
    return db_user

@router.get("/users", response_model=List[User])
async def read_users(skip: int = 0, limit: int = 100, current_user: User = Depends(get_current_user)):
    """
    Description:
        Получение списка пользователей.

    Args:
        skip: Количество пропускаемых пользователей.
        limit: Максимальное количество возвращаемых пользователей.
        current_user: Текущий аутентифицированный пользователь.

    Returns:
        Список пользователей.

    Raises:
        HTTPException: Если пользователь не аутентифицирован.

    Examples:
        >>> response = client.get("/users", headers={"Authorization": f"Bearer {token}"})
        >>> response.status_code
        200
        >>> len(response.json()) > 0
        True
    """
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    users = list(db.users.values())[skip : skip + limit]
    return users

@router.get("/users/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_user)):
    """
    Description:
        Получение информации о текущем аутентифицированном пользователе.

    Args:
        current_user: Текущий аутентифицированный пользователь.

    Returns:
        Объект текущего пользователя.

    Raises:
        HTTPException: Если пользователь не аутентифицирован.

    Examples:
        >>> response = client.get("/users/me", headers={"Authorization": f"Bearer {token}"})
        >>> response.status_code
        200
        >>> response.json()["username"] == "testuser"
        True
    """
    if current_user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    user = db.get(UserDB, current_user.id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return current_user

@router.get("/users/{user_id}", response_model=User)
async def read_user(user_id: UUID, current_user: User = Depends(get_current_user)):
    """
    Description:
        Получение информации о конкретном пользователе.

    Args:
        user_id: ID пользователя.
        current_user: Текущий аутентифицированный пользователь.

    Returns:
        Объект пользователя.

    Raises:
        HTTPException: Если пользователь не найден или не аутентифицирован.

    Examples:
        >>> user = db.create_user(UserCreate(username="testuser", email="test@example.com", password="testpass"))
        >>> response = client.get(f"/users/{user.id}", headers={"Authorization": f"Bearer {token}"})
        >>> response.status_code
        200
        >>> response.json()["username"]
        'testuser'
    """
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    user = db.get(UserDB, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@router.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Description:
        Аутентификация пользователя и выдача токена доступа.

    Args:
        form_data: Данные формы для аутентификации.

    Returns:
        Токен доступа.

    Raises:
        HTTPException: Если аутентификация не удалась.

    Examples:
        >>> response = client.post("/token", data={"username": "testuser", "password": "testpass"})
        >>> response.status_code
        200
        >>> "access_token" in response.json()
        True
    """
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}
