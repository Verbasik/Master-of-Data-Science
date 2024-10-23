# project-management-system/services/user_service/src/auth.py
# ============================
# БЛОК ИМПОРТОВ
# ============================

# Импорты из стандартной библиотеки
# Эти модули предоставляют базовые возможности для работы с датой, временем и типами данных.
from datetime import datetime, timedelta  # Работа с датой и временем
from typing import Optional               # Опциональные типы для аннотаций

# Импорты из сторонних библиотек
# Эти модули нужны для реализации API, безопасности и токенов.
from fastapi import Depends, HTTPException, status        # Функции для зависимостей и обработки ошибок
from fastapi.security import OAuth2PasswordBearer         # Авторизация с использованием токенов OAuth2
from jose import JWTError, jwt                            # Работа с JWT-токенами
from passlib.context import CryptContext                  # Контекст для хеширования паролей

# Импорты внутренних модулей проекта
# Эти модули реализуют модели данных и доступ к базе данных.
from models import UserInDB, Token  # Модели пользователя и токена
from database import db             # Модуль для работы с базой данных


# Настройки для JWT
SECRET_KEY = "your-secret-key"  # В реальном приложении должен храниться в переменных окружения
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Контекст для хеширования паролей
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Description:
      Проверка соответствия пароля его хешу.

    Args:
        plain_password: Пароль в открытом виде.
        hashed_password: Хешированный пароль.

    Returns:
        True, если пароль соответствует хешу, иначе False.

    Examples:
        >>> hashed = get_password_hash("mypassword")
        >>> verify_password("mypassword", hashed)
        True
        >>> verify_password("wrongpassword", hashed)
        False
    """
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """
    Description:
      Получение хеша пароля.

    Args:
        password: Пароль в открытом виде.

    Returns:
        Хешированный пароль.

    Examples:
        >>> hashed = get_password_hash("mypassword")
        >>> len(hashed) > 0
        True
    """
    return pwd_context.hash(password)

def authenticate_user(username: str, password: str) -> Optional[UserInDB]:
    """
    Description:
      Аутентификация пользователя.

    Args:
        username: Имя пользователя.
        password: Пароль пользователя.

    Returns:
        Объект пользователя, если аутентификация успешна, иначе None.

    Examples:
        >>> user = UserInDB(username="testuser", email="test@example.com", hashed_password=get_password_hash("testpass"))
        >>> db.create_user(user)
        >>> authenticate_user("testuser", "testpass") is not None
        True
        >>> authenticate_user("testuser", "wrongpass") is None
        True
    """
    user = db.get_user_by_username(username)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Description:
      Создание JWT токена.

    Args:
        data: Данные для включения в токен.
        expires_delta: Время жизни токена.

    Returns:
        Строка с JWT токеном.

    Examples:
        >>> token = create_access_token({"sub": "testuser"})
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

def get_current_user(token: str = Depends(oauth2_scheme)) -> Optional[UserInDB]:
    """
    Description:
      Получение текущего пользователя по токену.

    Args:
        token: JWT токен.

    Returns:
        Объект пользователя, если токен валиден, иначе None.

    Raises:
        JWTError: Если токен невалиден.

    Examples:
        >>> user = UserInDB(username="testuser", email="test@example.com", hashed_password=get_password_hash("testpass"))
        >>> db.create_user(user)
        >>> token = create_access_token({"sub": user.username})
        >>> get_current_user(token) is not None
        True
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
        user = db.get_user_by_username(username)
        if user is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
        return user
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")