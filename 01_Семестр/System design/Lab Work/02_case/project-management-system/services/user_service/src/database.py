# project-management-system/services/user_service/src/database.py
# ============================
# БЛОК ИМПОРТОВ
# ============================
# Импорты из стандартной библиотеки
# Используются для работы с аннотациями типов и UUID.
from typing import Dict, Optional  # Словари и опциональные типы для аннотаций
from uuid import UUID              # Обработка и использование UUID

# Импорты внутренних модулей проекта
# Эти модули реализуют модели данных.
from models import UserInDB        # Модель пользователя, хранящаяся в БД


class Database:
    """
    Description:
      Класс для работы с in-memory хранилищем пользователей.

    Attributes:
        users (Dict[UUID, UserInDB]): Словарь для хранения пользователей.

    Examples:
        >>> db = Database()
        >>> user = UserInDB(username="john_doe", email="john@example.com", hashed_password="hashedpass")
        >>> db.create_user(user)
        >>> db.get_user_by_username("john_doe")
        UserInDB(id=UUID('...'), username='john_doe', email='john@example.com', is_active=True, hashed_password='hashedpass')
    """

    def __init__(self):
        """
        Description:
          Инициализация базы данных.
        """
        self.users: Dict[UUID, UserInDB] = {}

    def create_user(self, user: UserInDB) -> UserInDB:
        """
        Description:
          Создание нового пользователя в базе данных.

        Args:
            user: Объект пользователя для создания.

        Returns:
            Созданный объект пользователя.

        Raises:
            ValueError: Если пользователь с таким username уже существует.

        Examples:
            >>> db = Database()
            >>> user = UserInDB(username="john_doe", email="john@example.com", hashed_password="hashedpass")
            >>> db.create_user(user)
            UserInDB(id=UUID('...'), username='john_doe', email='john@example.com', is_active=True, hashed_password='hashedpass')
        """
        if self.get_user_by_username(user.username):
            raise ValueError(f"User with username {user.username} already exists")
        self.users[user.id] = user
        return user

    def get_user_by_id(self, user_id: UUID) -> Optional[UserInDB]:
        """
        Description:
          Получение пользователя по ID.

        Args:
            user_id: ID пользователя.

        Returns:
            Объект пользователя или None, если пользователь не найден.

        Examples:
            >>> db = Database()
            >>> user = UserInDB(username="john_doe", email="john@example.com", hashed_password="hashedpass")
            >>> created_user = db.create_user(user)
            >>> db.get_user_by_id(created_user.id)
            UserInDB(id=UUID('...'), username='john_doe', email='john@example.com', is_active=True, hashed_password='hashedpass')
        """
        return self.users.get(user_id)

    def get_user_by_username(self, username: str) -> Optional[UserInDB]:
        """
        Description:
          Получение пользователя по имени пользователя.

        Args:
            username: Имя пользователя.

        Returns:
            Объект пользователя или None, если пользователь не найден.

        Examples:
            >>> db = Database()
            >>> user = UserInDB(username="john_doe", email="john@example.com", hashed_password="hashedpass")
            >>> db.create_user(user)
            >>> db.get_user_by_username("john_doe")
            UserInDB(id=UUID('...'), username='john_doe', email='john@example.com', is_active=True, hashed_password='hashedpass')
        """
        for user in self.users.values():
            if user.username == username:
                return user
        return None

# Создание экземпляра базы данных
db = Database()

# Создание мастер-пользователя
master_user = UserInDB(username="admin", email="admin@example.com", hashed_password="hashed_secret")
db.create_user(master_user)