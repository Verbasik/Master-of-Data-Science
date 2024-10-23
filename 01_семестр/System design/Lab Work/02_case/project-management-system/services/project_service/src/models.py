# project-management-system/services/project_service/src/models.py
from pydantic import BaseModel, Field
from typing import Optional
from uuid import UUID, uuid4
from datetime import datetime

class ProjectBase(BaseModel):
    """
    Описание:
      Базовая модель проекта.

    Атрибуты:
        name (str): Название проекта.
        description (str): Описание проекта.

    Пример использования:
        >>> project = ProjectBase(name="Новый проект", description="Описание проекта")
        >>> project.dict()
        {'name': 'Новый проект', 'description': 'Описание проекта'}
    """
    name: str = Field(..., min_length=1, max_length=100)
    description: str = Field(..., max_length=1000)

class ProjectCreate(ProjectBase):
    """
    Описание:
      Модель для создания проекта.

    Атрибуты:
        name (str): Название проекта.
        description (str): Описание проекта.

    Пример использования:
        >>> project = ProjectCreate(name="Новый проект", description="Описание проекта")
        >>> project.dict()
        {'name': 'Новый проект', 'description': 'Описание проекта'}
    """
    pass

class Project(ProjectBase):
    """
    Описание:
      Полная модель проекта.

    Атрибуты:
        id (UUID): Уникальный идентификатор проекта.
        name (str): Название проекта.
        description (str): Описание проекта.
        created_at (datetime): Дата и время создания проекта.
        updated_at (datetime): Дата и время последнего обновления проекта.

    Пример использования:
        >>> project = Project(id=uuid4(), name="Новый проект", description="Описание проекта", 
        ...                   created_at=datetime.now(), updated_at=datetime.now())
        >>> project.dict()
        {'id': UUID('...'), 'name': 'Новый проект', 'description': 'Описание проекта', 
         'created_at': datetime(...), 'updated_at': datetime(...)}
    """
    id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

class ProjectInDB(Project):
    """
    Описание:
      Модель проекта для хранения в базе данных.

    Атрибуты:
        id (UUID): Уникальный идентификатор проекта.
        name (str): Название проекта.
        description (str): Описание проекта.
        created_at (datetime): Дата и время создания проекта.
        updated_at (datetime): Дата и время последнего обновления проекта.
        owner_id (UUID): Идентификатор владельца проекта.

    Пример использования:
        >>> project = ProjectInDB(id=uuid4(), name="Новый проект", description="Описание проекта", 
        ...                       created_at=datetime.now(), updated_at=datetime.now(), owner_id=uuid4())
        >>> project.dict()
        {'id': UUID('...'), 'name': 'Новый проект', 'description': 'Описание проекта', 
         'created_at': datetime(...), 'updated_at': datetime(...), 'owner_id': UUID('...')}
    """
    owner_id: UUID