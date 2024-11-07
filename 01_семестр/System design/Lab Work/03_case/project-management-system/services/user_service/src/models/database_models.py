# project-management-system/services/user_service/src/models/database_models.py
import uuid
from utils.database import Base

from sqlalchemy import Column, String, Boolean, DateTime
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func


class User(Base):
    """
    Description:
        Модель пользователя для базы данных.

    Attributes:
        id (UUID): Уникальный идентификатор пользователя.
        username (str): Уникальное имя пользователя, не может быть пустым.
        email (str): Уникальный адрес электронной почты пользователя, не может быть пустым.
        hashed_password (str): Хэшированный пароль пользователя, не может быть пустым.
        is_active (bool): Статус активности пользователя. По умолчанию True.
        is_admin (bool): Статус администратора пользователя. По умолчанию False.
        created_at (datetime): Дата и время создания записи о пользователе.
        updated_at (datetime): Дата и время последнего обновления записи о пользователе.
    """

    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
