# project-management-system/services/project_service/src/database.py
from typing import Dict, List, Optional
from uuid import UUID
from models import ProjectInDB

class Database:
    """
    Описание:
      Класс для работы с in-memory хранилищем проектов.

    Атрибуты:
        projects (Dict[UUID, ProjectInDB]): Словарь для хранения проектов.

    Пример использования:
        >>> db = Database()
        >>> project = ProjectInDB(name="Тестовый проект", description="Описание", owner_id=UUID('...'))
        >>> db.create_project(project)
        >>> db.get_project_by_id(project.id)
        ProjectInDB(id=UUID('...'), name='Тестовый проект', description='Описание', ...)
    """

    def __init__(self):
        """
        Описание:
          Инициализация базы данных.
        """
        self.projects: Dict[UUID, ProjectInDB] = {}

    def create_project(self, project: ProjectInDB) -> ProjectInDB:
        """
        Описание:
          Создание нового проекта в базе данных.

        Аргументы:
            project (ProjectInDB): Объект проекта для создания.

        Возвращает:
            ProjectInDB: Созданный объект проекта.

        Исключения:
            ValueError: Если проект с таким id уже существует.

        Пример использования:
            >>> db = Database()
            >>> project = ProjectInDB(name="Новый проект", description="Описание", owner_id=UUID('...'))
            >>> db.create_project(project)
            ProjectInDB(id=UUID('...'), name='Новый проект', description='Описание', ...)
        """
        if project.id in self.projects:
            raise ValueError(f"Project with id {project.id} already exists")
        self.projects[project.id] = project
        return project

    def get_project_by_id(self, project_id: UUID) -> Optional[ProjectInDB]:
        """
        Описание:
          Получение проекта по ID.

        Аргументы:
            project_id (UUID): ID проекта.

        Возвращает:
            Optional[ProjectInDB]: Объект проекта или None, если проект не найден.

        Пример использования:
            >>> db = Database()
            >>> project = ProjectInDB(name="Тестовый проект", description="Описание", owner_id=UUID('...'))
            >>> created_project = db.create_project(project)
            >>> db.get_project_by_id(created_project.id)
            ProjectInDB(id=UUID('...'), name='Тестовый проект', description='Описание', ...)
        """
        return self.projects.get(project_id)

    def get_projects_by_owner(self, owner_id: UUID) -> List[ProjectInDB]:
        """
        Описание:
          Получение всех проектов пользователя.

        Аргументы:
            owner_id (UUID): ID владельца проектов.

        Возвращает:
            List[ProjectInDB]: Список проектов пользователя.

        Пример использования:
            >>> db = Database()
            >>> owner_id = UUID('...')
            >>> project1 = ProjectInDB(name="Проект 1", description="Описание 1", owner_id=owner_id)
            >>> project2 = ProjectInDB(name="Проект 2", description="Описание 2", owner_id=owner_id)
            >>> db.create_project(project1)
            >>> db.create_project(project2)
            >>> db.get_projects_by_owner(owner_id)
            [ProjectInDB(...), ProjectInDB(...)]
        """
        return [project for project in self.projects.values() if project.owner_id == owner_id]

    def update_project(self, project: ProjectInDB) -> ProjectInDB:
        """
        Описание:
          Обновление существующего проекта.

        Аргументы:
            project (ProjectInDB): Обновленный объект проекта.

        Возвращает:
            ProjectInDB: Обновленный объект проекта.

        Исключения:
            ValueError: Если проект с указанным id не существует.

        Пример использования:
            >>> db = Database()
            >>> project = ProjectInDB(name="Старое название", description="Старое описание", owner_id=UUID('...'))
            >>> created_project = db.create_project(project)
            >>> created_project.name = "Новое название"
            >>> db.update_project(created_project)
            ProjectInDB(id=UUID('...'), name='Новое название', description='Старое описание', ...)
        """
        if project.id not in self.projects:
            raise ValueError(f"Project with id {project.id} does not exist")
        self.projects[project.id] = project
        return project

    def delete_project(self, project_id: UUID) -> None:
        """
        Описание:
          Удаление проекта по ID.

        Аргументы:
            project_id (UUID): ID проекта для удаления.

        Исключения:
            ValueError: Если проект с указанным id не существует.

        Пример использования:
            >>> db = Database()
            >>> project = ProjectInDB(name="Проект для удаления", description="Описание", owner_id=UUID('...'))
            >>> created_project = db.create_project(project)
            >>> db.delete_project(created_project.id)
        """
        if project_id not in self.projects:
            raise ValueError(f"Project with id {project_id} does not exist")
        del self.projects[project_id]

# Создание экземпляра базы данных
db = Database()