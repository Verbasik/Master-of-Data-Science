# project-management-system/services/project_service/src/models/database_models.py
import uuid
from utils.database import Base

from sqlalchemy import Column, String, Text, DateTime
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func

class Project(Base):
    """
    Description:
        Модель для представления проекта в базе данных.

    Attributes:
        id: Уникальный идентификатор проекта (UUID).
        name: Название проекта, обязательное поле (String).
        description: Описание проекта (Text).
        owner_id: Идентификатор владельца проекта (UUID), обязательное поле.
        created_at: Дата и время создания проекта с установкой текущего времени по умолчанию.
        updated_at: Дата и время последнего обновления с автообновлением при изменениях.
    """
    __tablename__ = "projects"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    owner_id = Column(UUID(as_uuid=True), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), 
        server_default=func.now(), 
        onupdate=func.now()
    )

    def __repr__(self) -> str:
        """
        Description:
            Возвращает строковое представление объекта Project для отладки.

        Returns:
            str: Строка с информацией о проекте.
        """
        return f"<Project(id={self.id}, name={self.name})>"