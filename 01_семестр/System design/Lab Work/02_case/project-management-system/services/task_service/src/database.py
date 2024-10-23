# project-management-system/services/task_service/src/database.py
from typing import Dict, List, Optional
from uuid import UUID
from models import Task
from datetime import datetime

class Database:
    """
    Описание:
      Класс для работы с in-memory хранилищем задач.

    Атрибуты:
        tasks: Словарь для хранения задач
    """
    def __init__(self):
        self.tasks: Dict[UUID, Task] = {}

    def create_task(self, task: Task) -> Task:
        """
        Описание:
          Создание новой задачи.

        Args:
            task: Объект задачи для создания

        Returns:
            Task: Созданная задача

        Raises:
            ValueError: Если задача с таким ID уже существует
        """
        if task.id in self.tasks:
            raise ValueError(f"Task with id {task.id} already exists")
        self.tasks[task.id] = task
        return task

    def get_task_by_id(self, task_id: UUID) -> Optional[Task]:
        """
        Описание:
          Получение задачи по ID.

        Args:
            task_id: ID задачи

        Returns:
            Optional[Task]: Найденная задача или None
        """
        return self.tasks.get(task_id)

    def get_tasks_by_project(self, project_id: UUID) -> List[Task]:
        """
        Описание:
          Получение всех задач проекта.

        Args:
            project_id: ID проекта

        Returns:
            List[Task]: Список задач проекта
        """
        return [task for task in self.tasks.values() if task.project_id == project_id]

    def get_tasks_by_assignee(self, assignee_id: UUID) -> List[Task]:
        """
        Описание:
          Получение всех задач исполнителя.

        Args:
            assignee_id: ID исполнителя

        Returns:
            List[Task]: Список задач исполнителя
        """
        return [task for task in self.tasks.values() if task.assignee_id == assignee_id]

    def update_task(self, task_id: UUID, update_data: dict) -> Optional[Task]:
        """
        Описание:
          Обновление задачи.

        Args:
            task_id: ID задачи
            update_data: Словарь с обновляемыми полями

        Returns:
            Optional[Task]: Обновленная задача или None

        Raises:
            ValueError: Если задача не найдена
        """
        if task_id not in self.tasks:
            raise ValueError(f"Task with id {task_id} not found")
        
        task = self.tasks[task_id]
        updated_task_dict = task.dict()
        updated_task_dict.update(update_data)
        updated_task_dict["updated_at"] = datetime.now()
        
        self.tasks[task_id] = Task(**updated_task_dict)
        return self.tasks[task_id]

    def delete_task(self, task_id: UUID) -> bool:
        """
        Описание:
          Удаление задачи.

        Args:
            task_id: ID задачи

        Returns:
            bool: True если задача удалена, False если задача не найдена
        """
        if task_id in self.tasks:
            del self.tasks[task_id]
            return True
        return False

# Создание экземпляра базы данных
db = Database()