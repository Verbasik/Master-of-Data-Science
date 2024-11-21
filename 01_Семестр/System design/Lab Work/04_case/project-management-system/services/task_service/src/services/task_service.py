# project-management-system/services/task_service/src/services/task_service.py
# Импорт стандартных библиотек
import logging
from uuid import UUID
from typing import List, Optional, Dict, Any

# Project imports
from ..models.mongo_models import MongoTask
from ..models.api_models import TaskCreate, TaskUpdate

# Настройка логирования
logger = logging.getLogger(__name__)

class TaskService:
    """
    Description:
        Сервис для управления задачами в проекте.
    """
    
    def __init__(self, db_manager):
        """
        Description:
            Инициализация TaskService с менеджером базы данных.

        Args:
            db_manager: Менеджер для взаимодействия с базой данных.
        """
        self.db = db_manager

    async def create_task(self, task_data: TaskCreate, creator_id: UUID) -> MongoTask:
        """
        Description:
            Создает новую задачу и сохраняет её в базе данных.

        Args:
            task_data: Данные для создания задачи.
            creator_id: Идентификатор пользователя, создающего задачу.

        Returns:
            Созданная задача в формате MongoTask.

        Raises:
            Exception: В случае ошибки сохранения задачи.

        Examples:
            >>> await create_task(task_data, UUID("123e4567-e89b-12d3-a456-426614174000"))
        """
        try:
            logger.info("Начало создания задачи в TaskService")
            logger.debug(f"Данные задачи: {task_data.model_dump_json(indent=2)}")
            logger.debug(f"ID создателя: {creator_id}")

            task = MongoTask(
                title=task_data.title,
                description=task_data.description,
                priority=task_data.priority,
                project_id=task_data.project_id,
                creator_id=creator_id,
                assignee_id=task_data.assignee_id
            )
            logger.debug(f"Создан объект MongoTask: {task.model_dump_json(indent=2)}")

            # Сохранение задачи в базе данных
            try:
                saved_task = await self.db.create_task(task)
                logger.info(f"Задача успешно сохранена: {saved_task.model_dump_json(indent=2)}")
                return saved_task
            except Exception as e:
                logger.error(f"Ошибка сохранения задачи в MongoDB: {str(e)}")
                raise Exception("Ошибка сохранения задачи в базу данных") from e

        except Exception as e:
            logger.error(f"Ошибка в TaskService.create_task: {str(e)}")
            raise

    async def update_task(self, task_id: UUID, task_update: TaskUpdate) -> Optional[MongoTask]:
        """
        Description:
            Обновляет данные задачи.

        Args:
            task_id: Идентификатор задачи.
            task_update: Данные для обновления задачи.

        Returns:
            Обновлённая задача или None, если задача не найдена.

        Examples:
            >>> await update_task(UUID("123e4567-e89b-12d3-a456-426614174000"), task_update)
        """
        try:
            update_data = task_update.dict(exclude_unset=True)
            updated_task = await self.db.update_task(task_id, update_data)
            logger.info(f"Задача {task_id} успешно обновлена.")
            return updated_task
        except Exception as e:
            logger.error(f"Ошибка обновления задачи: {str(e)}")
            raise

    async def delete_task(self, task_id: UUID) -> bool:
        """
        Description:
            Удаляет задачу из базы данных.

        Args:
            task_id: Идентификатор задачи.

        Returns:
            True, если задача успешно удалена, иначе False.

        Examples:
            >>> await delete_task(UUID("123e4567-e89b-12d3-a456-426614174000"))
        """
        try:
            is_deleted = await self.db.delete_task(task_id)
            if is_deleted:
                logger.info(f"Задача {task_id} успешно удалена.")
            else:
                logger.warning(f"Задача {task_id} не найдена для удаления.")
            return is_deleted
        except Exception as e:
            logger.error(f"Ошибка удаления задачи: {str(e)}")
            raise

    async def get_project_tasks(self, project_id: UUID) -> List[MongoTask]:
        """
        Description:
            Получает список задач проекта.

        Args:
            project_id: Идентификатор проекта.

        Returns:
            Список задач проекта.

        Examples:
            >>> await get_project_tasks(UUID("123e4567-e89b-12d3-a456-426614174000"))
        """
        try:
            tasks = await self.db.get_tasks_by_project(project_id)
            logger.info(f"Найдено {len(tasks)} задач для проекта {project_id}.")
            return tasks
        except Exception as e:
            logger.error(f"Ошибка получения задач проекта: {str(e)}")
            raise

    async def get_tasks_by_criteria(self, criteria: Dict[str, Any]) -> List[MongoTask]:
        """
        Description:
            Получает задачи, соответствующие заданным критериям.

        Args:
            criteria: Словарь с критериями фильтрации задач.

        Returns:
            Список задач, соответствующих критериям.

        Examples:
            >>> await get_tasks_by_criteria({"priority": "high"})
        """
        try:
            tasks = await self.db.get_tasks_by_criteria(criteria)
            logger.info(f"Найдено {len(tasks)} задач по критериям {criteria}.")
            return tasks
        except Exception as e:
            logger.error(f"Ошибка получения задач по критериям: {str(e)}")
            raise
