# project-management-system/services/project_service/src/models/api_models.py
from pydantic import BaseModel, Field
from typing import Optional
from uuid import UUID, uuid4
from datetime import datetime

class ProjectBase(BaseModel):
    """
    Description:
        Базовая модель проекта с основными полями, такими как
        название и описание.

    Attributes:
        name: Название проекта
        description: Описание проекта
    """
    name: str = Field(..., min_length=1, max_length=100)
    description: str = Field(..., max_length=1000)

class ProjectCreate(ProjectBase):
    """
    Description:
        Модель для создания проекта, наследуется от ProjectBase.
    """
    pass

class Project(ProjectBase):
    """
    Description:
        Полная модель проекта с идентификатором, датами создания
        и обновления, а также количеством задач.

    Attributes:
        id: Уникальный идентификатор проекта
        owner_id: Идентификатор владельца проекта
        created_at: Дата и время создания проекта
        updated_at: Дата и время последнего обновления проекта
        task_count: Опциональное количество задач в проекте
    """
    id: UUID = Field(default_factory=uuid4)
    owner_id: UUID
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    task_count: Optional[int] = None

    class Config:
        """
        Config:
            Настройки конфигурации для использования атрибутов.
        """
        from_attributes = True

class ProjectInDB(Project):
    """
    Description:
        Модель проекта для хранения в базе данных, наследуется от Project.
    """
    pass

class ProjectUpdate(BaseModel):
    """
    Description:
        Модель для обновления проекта с опциональными полями.

    Attributes:
        name: Опциональное новое название проекта
        description: Опциональное новое описание проекта
    """
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=1000)

class ProjectResponse(Project):
    """
    Description:
        Модель ответа для API, расширяет Project, убирая избыточные
        зависимости.

    Attributes:
        id: Уникальный идентификатор проекта
        owner_id: Идентификатор владельца проекта
        created_at: Дата и время создания проекта
        updated_at: Дата и время последнего обновления проекта
    """
    id: UUID = Field(default_factory=uuid4)
    owner_id: UUID
    created_at: datetime
    updated_at: datetime

    class Config:
        """
        Config:
            Настройки конфигурации для использования атрибутов.
        """
        from_attributes = True
