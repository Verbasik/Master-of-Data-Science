# project-management-system/services/task_service/src/database.py

# Испорт стандартных библиотек
import logging
from datetime import datetime
from uuid import UUID
from typing import Optional, List, Dict, Any

# Импорт сторонних библиотек
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import IndexModel, ASCENDING, DESCENDING
from bson import UuidRepresentation

# Импорт библиотек проекта
from utils.database import db_manager
from .models.mongo_models import MongoTask
from .core.config import MONGODB_URL, MONGODB_DB_NAME

# Logging setup
logger = logging.getLogger(__name__)

class MongoDBManager:
    """
    Description:
        Класс для управления соединением и взаимодействием с MongoDB.
    """
    def __init__(self):
        """
        Description:
            Инициализация менеджера MongoDB.
        """
        self.client: Optional[AsyncIOMotorClient] = None
        self.db = None
        self.tasks = None

    async def connect(self) -> None:
        """
        Description:
            Подключается к базе данных MongoDB и создает необходимые индексы.

        Raises:
            Exception: В случае ошибки подключения.
        """
        try:
            self.client = AsyncIOMotorClient(
                MONGODB_URL,
                uuidRepresentation='standard'
            )
            self.db = self.client[MONGODB_DB_NAME]
            self.tasks = self.db.tasks
            await self.ensure_indexes()
            logger.info("Подключение к MongoDB успешно выполнено.")
            return self  # Возвращаем self для цепочки вызовов
        except Exception as e:
            logger.error(f"Ошибка подключения к MongoDB: {e}")
            raise

    async def disconnect(self) -> None:
        """
        Description:
            Закрывает соединение с базой данных.
        """
        if self.client:
            self.client.close()
            logger.info("Соединение с MongoDB закрыто.")

    async def ensure_indexes(self) -> None:
        """
        Description:
            Создает индексы для коллекции задач.
        """
        try:
            indexes = [
                IndexModel([("project_id", ASCENDING)]),
                IndexModel([("creator_id", ASCENDING)]),
                IndexModel([("assignee_id", ASCENDING)]),
                IndexModel([("status", ASCENDING)]),
                IndexModel([("priority", ASCENDING)]),
                IndexModel([
                    ("project_id", ASCENDING),
                    ("status", ASCENDING),
                    ("priority", DESCENDING)
                ]),
                IndexModel([("created_at", DESCENDING)]),
                IndexModel([("tags", ASCENDING)])
            ]
            await self.tasks.create_indexes(indexes)
            logger.info("Индексы успешно созданы.")
        except Exception as e:
            logger.error(f"Ошибка создания индексов: {e}")
            raise

    async def create_task(self, task: MongoTask) -> MongoTask:
        """
        Description:
            Создает новую задачу и сохраняет её в базе данных.

        Args:
            task: Экземпляр MongoTask, содержащий данные задачи.

        Returns:
            Созданная задача в формате MongoTask.

        Raises:
            Exception: В случае ошибки создания задачи.
        """
        try:
            task_dict = task.model_dump(by_alias=True)
            logger.debug(f"Преобразование MongoTask в словарь: {task_dict}")

            mongo_dict = self._convert_uuids_to_str(task_dict)
            result = await self.tasks.insert_one(mongo_dict)

            saved_doc = await self.tasks.find_one({"_id": result.inserted_id})
            if not saved_doc:
                raise Exception("Документ не найден после вставки.")

            saved_task = MongoTask(**self._convert_ids_to_uuid(saved_doc))
            logger.info("Задача успешно создана.")
            return saved_task
        except Exception as e:
            logger.error(f"Ошибка создания задачи: {e}")
            raise

    async def get_task(self, task_id: UUID) -> Optional[MongoTask]:
        """
        Description:
            Получает задачу по её идентификатору.

        Args:
            task_id: Идентификатор задачи.

        Returns:
            Экземпляр MongoTask или None, если задача не найдена.

        Raises:
            Exception: В случае ошибки поиска.
        """
        try:
            task_dict = await self.tasks.find_one({"_id": str(task_id)})
            if task_dict:
                return MongoTask(**self._convert_ids_to_uuid(task_dict))
            return None
        except Exception as e:
            logger.error(f"Ошибка получения задачи: {e}")
            raise

    async def update_task(self, task_id: UUID, update_data: Dict[str, Any]) -> Optional[MongoTask]:
        """
        Description:
            Обновляет задачу в базе данных.

        Args:
            task_id: Идентификатор задачи.
            update_data: Словарь с данными для обновления.

        Returns:
            Обновлённая задача или None, если задача не найдена.

        Raises:
            Exception: В случае ошибки обновления.
        """
        try:
            update_data["updated_at"] = datetime.utcnow()
            update_data = self._convert_uuids_to_str(update_data)

            result = await self.tasks.update_one(
                {"_id": str(task_id)},
                {"$set": update_data}
            )
            if result.modified_count:
                return await self.get_task(task_id)
            return None
        except Exception as e:
            logger.error(f"Ошибка обновления задачи: {e}")
            raise

    async def delete_task(self, task_id: UUID) -> bool:
        """
        Description:
            Удаляет задачу из базы данных.

        Args:
            task_id: Идентификатор задачи.

        Returns:
            True, если задача успешно удалена, иначе False.

        Raises:
            Exception: В случае ошибки удаления.
        """
        try:
            result = await self.tasks.delete_one({"_id": str(task_id)})
            return result.deleted_count > 0
        except Exception as e:
            logger.error(f"Ошибка удаления задачи: {e}")
            raise

    def _convert_uuids_to_str(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Description:
            Конвертирует UUID в строки в словаре.

        Args:
            data: Входной словарь.

        Returns:
            Словарь с преобразованными значениями.
        """
        return {
            key: (str(value) if isinstance(value, UUID) else value)
            for key, value in data.items()
        }

    def _convert_ids_to_uuid(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Description:
            Конвертирует строковые идентификаторы в UUID.

        Args:
            data: Словарь данных.

        Returns:
            Словарь с преобразованными идентификаторами.
        """
        for field in ["_id", "project_id", "creator_id", "assignee_id"]:
            if field in data and data[field]:
                data[field] = UUID(data[field])
        return data
    
# Реэкспорт db_manager для обратной совместимости
db = db_manager