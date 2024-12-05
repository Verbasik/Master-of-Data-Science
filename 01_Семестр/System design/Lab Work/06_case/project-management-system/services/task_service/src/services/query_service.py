# project-management-system/services/task_service/src/services/query_service.py

# Стандартные библиотеки
import logging
from typing import List, Optional, Dict, Any
from uuid import UUID

# Локальные импорты
from ..database import MongoDBManager
from ..models.mongo_models import MongoTask

# Настройка логирования
logger = logging.getLogger(__name__)


class TaskQueryService:
    """
    Description:
        Сервис для выполнения запросов к задачам в MongoDB.

    Args:
        db_manager: Менеджер базы данных MongoDB

    Examples:
        >>> service = TaskQueryService(db_manager)
        >>> task = await service.get_task(task_id)
    """

    def __init__(self, db_manager: MongoDBManager) -> None:
        self.db: MongoDBManager = db_manager

    async def get_task(self, task_id: UUID) -> Optional[MongoTask]:
        """
        Description:
            Получает задачу по идентификатору.

        Args:
            task_id: Идентификатор задачи

        Returns:
            Optional[MongoTask]: Найденная задача или None

        Raises:
            Exception: При ошибке получения задачи

        Examples:
            >>> task = await service.get_task(UUID("123..."))
        """
        try:
            return await self.db.get_task(task_id)
        except Exception as e:
            logger.error(f"Error in get_task query: {str(e)}")
            raise

    async def get_tasks_by_criteria(
        self,
        criteria: Dict[str, Any]
    ) -> List[MongoTask]:
        """
        Description:
            Получает список задач по заданным критериям.

        Args:
            criteria: Словарь критериев фильтрации

        Returns:
            List[MongoTask]: Список найденных задач

        Raises:
            Exception: При ошибке получения задач

        Examples:
            >>> tasks = await service.get_tasks_by_criteria({
            ...     "status": "IN_PROGRESS"
            ... })
        """
        try:
            return await self.db.get_tasks_by_criteria(criteria)
        except Exception as e:
            logger.error(f"Error in get_tasks_by_criteria query: {str(e)}")
            raise