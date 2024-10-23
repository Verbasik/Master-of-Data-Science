# project-management-system/services/task_service/src/models.py
from pydantic import BaseModel, Field
from typing import Optional
from uuid import UUID, uuid4
from datetime import datetime
from enum import Enum

class TaskStatus(str, Enum):
    """
    Описание:
      Перечисление возможных статусов задачи.

    Values:
        CREATED: Задача создана
        IN_PROGRESS: Задача в работе
        ON_REVIEW: Задача на проверке
        COMPLETED: Задача завершена
    """
    CREATED = "created"
    IN_PROGRESS = "in_progress"
    ON_REVIEW = "on_review"
    COMPLETED = "completed"

class TaskPriority(str, Enum):
    """
    Описание:
      Перечисление возможных приоритетов задачи.

    Values:
        LOW: Низкий приоритет
        MEDIUM: Средний приоритет
        HIGH: Высокий приоритет
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class TaskBase(BaseModel):
    """
    Описание:
      Базовая модель задачи.

    Атрибуты:
        title: Название задачи
        description: Описание задачи
        priority: Приоритет задачи
    """
    title: str = Field(..., min_length=1, max_length=200)
    description: str = Field(..., max_length=2000)
    priority: TaskPriority = Field(default=TaskPriority.MEDIUM)

class TaskCreate(TaskBase):
    """
    Описание:
      Модель для создания задачи.

    Атрибуты:
        project_id: ID проекта
        assignee_id: ID исполнителя
    """
    project_id: UUID
    assignee_id: Optional[UUID] = None

class Task(TaskBase):
    """
    Описание:
      Полная модель задачи.

    Атрибуты:
        id: Уникальный идентификатор задачи
        status: Статус задачи
        created_at: Дата и время создания
        updated_at: Дата и время обновления
        project_id: ID проекта
        assignee_id: ID исполнителя
        creator_id: ID создателя
    """
    id: UUID = Field(default_factory=uuid4)
    status: TaskStatus = Field(default=TaskStatus.CREATED)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    project_id: UUID
    assignee_id: Optional[UUID]
    creator_id: UUID

class TaskUpdate(BaseModel):
    """
    Описание:
      Модель для обновления задачи.

    Атрибуты:
        title: Название задачи
        description: Описание задачи
        status: Статус задачи
        priority: Приоритет задачи
        assignee_id: ID исполнителя
    """
    title: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=2000)
    status: Optional[TaskStatus] = None
    priority: Optional[TaskPriority] = None
    assignee_id: Optional[UUID] = None