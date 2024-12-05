# project-management-system/services/task_service/src/models/database_models.py
import uuid
import enum
from utils.database import Base

from sqlalchemy import Column, String, Text, DateTime, ForeignKey, Enum as SQLAlchemyEnum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

class TaskStatus(str, enum.Enum):
    """
    Description:
        Enum, представляющий возможные статусы задачи.
    """
    CREATED     = "created"
    IN_PROGRESS = "in_progress"
    ON_REVIEW   = "on_review"
    COMPLETED   = "completed"

class TaskPriority(str, enum.Enum):
    """
    Description:
        Enum, представляющий приоритеты задачи.
    """
    LOW    = "low"
    MEDIUM = "medium"
    HIGH   = "high"

class Task(Base):
    """
    Description:
        Модель для представления задачи в системе управления проектами.

    Attributes:
        id (UUID): Уникальный идентификатор задачи.
        title (str): Название задачи, краткое описание.
        description (str): Полное описание задачи.
        status (TaskStatus): Статус задачи (например, создана, в процессе, на проверке, завершена).
        priority (TaskPriority): Приоритет задачи (низкий, средний, высокий).
        project_id (UUID): Внешний ключ к проекту, к которому относится задача.
        creator_id (UUID): Внешний ключ к создателю задачи (пользователь, который создал).
        assignee_id (UUID): Внешний ключ к исполнителю задачи (пользователь, который выполняет).
        created_at (DateTime): Время создания задачи.
        updated_at (DateTime): Время последнего обновления задачи.
    """
    
    __tablename__ = "tasks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(200), nullable=False)
    description = Column(Text)
    status = Column(SQLAlchemyEnum(TaskStatus), default=TaskStatus.CREATED, nullable=False)
    priority = Column(SQLAlchemyEnum(TaskPriority), default=TaskPriority.MEDIUM, nullable=False)
    
    project_id = Column(UUID(as_uuid=True), nullable=False)
    creator_id = Column(UUID(as_uuid=True), nullable=False)
    assignee_id = Column(UUID(as_uuid=True), nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    def __repr__(self) -> str:
        """
        Description:
            Возвращает строковое представление объекта Task.

        Returns:
            str: Строковое представление объекта Task с id и названием задачи.
        """
        return f"<Task(id={self.id}, title={self.title})>"
