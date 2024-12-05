# project-management-system/services/user_service/src/cache.py
import json
from typing import Optional, Union
from uuid import UUID
from datetime import datetime
import redis

# Импорты моделей и базы данных
from .models.api_models import User
from .models.database_models import User as UserDB
from .database import db


class RedisCache:
    """
    Description:
        Класс для работы с кешем пользователей в Redis. 
        Поддерживает методы получения, обновления и удаления данных пользователей.

    Args:
        host (str): Хост Redis-сервера
        port (int): Порт Redis-сервера
        password (str): Пароль для доступа к Redis
        db (int): Номер базы данных Redis (по умолчанию: 0)
        ttl (int): Время жизни записей в кеше в секундах (по умолчанию: 3600)
    """
    def __init__(self, host: str, port: int, password: str, db: int = 0, ttl: int = 3600):
        """
        Инициализация подключения к Redis.

        Args:
            host: Хост Redis-сервера
            port: Порт Redis-сервера
            password: Пароль для подключения
            db: Номер базы данных Redis (по умолчанию: 0)
            ttl: Время жизни кеша (по умолчанию: 3600 секунд)
        """
        self.redis = redis.Redis(
            host=host,
            port=port,
            password=password,
            db=db,
            decode_responses=True
        )
        self.ttl = ttl

    def _get_user_key(self, user_id: Union[str, UUID]) -> str:
        """
        Генерация ключа Redis для пользователя.

        Args:
            user_id: Идентификатор пользователя

        Returns:
            str: Ключ для хранения данных пользователя
        """
        return f"user:{str(user_id)}"

    def _serialize_user(self, user: Union[User, UserDB]) -> str:
        """
        Сериализация объекта пользователя в JSON-строку.

        Args:
            user: Объект пользователя (API или база данных)

        Returns:
            str: Сериализованные данные пользователя
        """
        user_dict = {
            "id": str(user.id),
            "username": user.username,
            "email": user.email,
            "is_active": user.is_active,
            "created_at": user.created_at.isoformat(),
            "updated_at": user.updated_at.isoformat()
        }
        return json.dumps(user_dict)

    def _deserialize_user(self, user_json: str) -> User:
        """
        Десериализация JSON-строки в объект User.

        Args:
            user_json: Сериализованные данные пользователя в формате JSON

        Returns:
            User: Десериализованный объект пользователя
        """
        user_dict = json.loads(user_json)
        user_dict["created_at"] = datetime.fromisoformat(user_dict["created_at"])
        user_dict["updated_at"] = datetime.fromisoformat(user_dict["updated_at"])
        return User(**user_dict)

    def get_user(self, user_id: Union[str, UUID]) -> Optional[User]:
        """
        Получить пользователя из кеша.

        Args:
            user_id: Идентификатор пользователя

        Returns:
            Optional[User]: Объект пользователя или None, если не найден
        """
        key = self._get_user_key(user_id)
        user_data = self.redis.get(key)
        return self._deserialize_user(user_data) if user_data else None

    def update_user(self, user: Union[User, UserDB]) -> None:
        """
        Обновить данные пользователя в кеше.

        Args:
            user: Объект пользователя (API или база данных)
        """
        key = self._get_user_key(str(user.id))
        self.redis.setex(key, self.ttl, self._serialize_user(user))

    def invalidate_user(self, user_id: Union[str, UUID]) -> None:
        """
        Удалить данные пользователя из кеша.

        Args:
            user_id: Идентификатор пользователя
        """
        key = self._get_user_key(str(user_id))
        self.redis.delete(key)
