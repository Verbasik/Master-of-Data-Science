# project-management-system/services/task_service/src/services/command_service.py

# Стандартные библиотеки
import logging
from typing import Dict, Any
from uuid import UUID, uuid4

# Локальные импорты
from ..events.kafka_producer import KafkaTaskProducer
from ..models.api_models import TaskCreate, TaskUpdate

# Настройка логирования
logger = logging.getLogger(__name__)


class TaskCommandService:
    """
    Description:
        Сервис для обработки команд, связанных с задачами.
        Управляет созданием и обновлением задач через события Kafka.

    Args:
        kafka_producer: Производитель событий Kafka

    Examples:
        >>> service = TaskCommandService(kafka_producer)
        >>> task = await service.create_task(task_data, creator_id)
    """

    def __init__(self, kafka_producer: KafkaTaskProducer) -> None:
        self.kafka_producer: KafkaTaskProducer = kafka_producer

    async def create_task(
        self,
        task_data: TaskCreate,
        creator_id: UUID
    ) -> Dict[str, Any]:
        """
        Description:
            Создает новую задачу и отправляет событие в Kafka.

        Args:
            task_data: Данные для создания задачи
            creator_id: Идентификатор создателя задачи

        Returns:
            Dict[str, Any]: Словарь с данными созданной задачи

        Raises:
            Exception: При ошибке создания задачи

        Examples:
            >>> task = await service.create_task(
            ...     TaskCreate(title="New Task"),
            ...     UUID("123...")
            ... )
        """
        try:
            task_dict = task_data.dict()
            task_dict["creator_id"] = creator_id
            task_dict["id"] = str(uuid4())
            
            await self.kafka_producer.send_task_created_event(task_dict)
            
            return task_dict
        except Exception as e:
            logger.error(f"Error in create_task command: {str(e)}")
            raise