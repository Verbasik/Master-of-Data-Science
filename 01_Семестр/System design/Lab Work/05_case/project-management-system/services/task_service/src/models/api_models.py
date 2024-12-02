# project-management-system/services/task_service/src/models/api_models.py
from typing import Optional
from uuid import UUID, uuid4
from datetime import datetime
from pydantic import BaseModel, Field
from .database_models import TaskStatus, TaskPriority

class TaskBase(BaseModel):
    """
    Description:
        Базовая модель задачи, содержащая основные поля задачи.

    Attributes:
        title (str): Заголовок задачи, от 1 до 200 символов.
        description (str): Описание задачи, до 2000 символов.
        priority (TaskPriority): Приоритет задачи (по умолчанию - MEDIUM).
    """
    title: str = Field(..., min_length=1, max_length=200)
    description: str = Field(..., max_length=2000)
    priority: TaskPriority = Field(default=TaskPriority.MEDIUM)

    class Config:
        from_attributes = True

class TaskCreate(TaskBase):
    """
    Description:
        Модель создания задачи, содержащая дополнительное поле project_id и необязательное поле assignee_id.

    Attributes:
        project_id (UUID): Идентификатор проекта, к которому относится задача.
        assignee_id (Optional[UUID]): Идентификатор пользователя, которому назначена задача.
    """
    project_id: UUID
    assignee_id: Optional[UUID] = None

class Task(TaskBase):
    """
    Description:
        Полная модель задачи, включающая все основные атрибуты, а также статус и временные метки.

    Attributes:
        id (UUID): Уникальный идентификатор задачи.
        status (TaskStatus): Статус задачи (по умолчанию - CREATED).
        project_id (UUID): Идентификатор проекта.
        creator_id (UUID): Идентификатор создателя задачи.
        assignee_id (Optional[UUID]): Идентификатор назначенного пользователя (если имеется).
        created_at (datetime): Дата и время создания задачи.
        updated_at (datetime): Дата и время последнего обновления задачи.

    Config:
        rom_attributes (bool): Разрешает работу с объектами ORM для Pydantic.
    """
    id: UUID = Field(default_factory=uuid4)
    status: TaskStatus = Field(default=TaskStatus.CREATED)
    project_id: UUID
    creator_id: UUID
    assignee_id: Optional[UUID]
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        rom_attributes = True

class TaskUpdate(BaseModel):
    """
    Description:
        Модель обновления задачи, включающая опциональные поля для обновления.

    Attributes:
        title (Optional[str]): Заголовок задачи, от 1 до 200 символов.
        description (Optional[str]): Описание задачи, до 2000 символов.
        status (Optional[TaskStatus]): Обновленный статус задачи.
        priority (Optional[TaskPriority]): Обновленный приоритет задачи.
        assignee_id (Optional[UUID]): Обновленный идентификатор назначенного пользователя.
    """
    title: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=2000)
    status: Optional[TaskStatus] = None
    priority: Optional[TaskPriority] = None
    assignee_id: Optional[UUID] = None

class TaskResponse(Task):
    """
    Description:
        Модель ответа задачи с дополнительными полями для удобства отображения.
    """

    @classmethod
    def from_mongo(cls, mongo_task: "MongoTask") -> "TaskResponse":
        """
        Преобразует объект MongoTask в модель TaskResponse.

        Args:
            mongo_task: Объект задачи из базы данных MongoDB.

        Returns:
            Экземпляр TaskResponse.

        Examples:
            >>> mongo_task = MongoTask(...)
            >>> task_response = TaskResponse.from_mongo(mongo_task)
        """
        return cls(
            id          = mongo_task.id,
            title       = mongo_task.title,
            description = mongo_task.description,
            status      = mongo_task.status,
            priority    = mongo_task.priority,
            project_id  = mongo_task.project_id,
            creator_id  = mongo_task.creator_id,
            assignee_id = mongo_task.assignee_id,
            created_at  = mongo_task.created_at,
            updated_at  = mongo_task.updated_at,
        )

    class Config:
        """
        Дополнительные настройки модели Pydantic.
        """
        from_attributes = True
