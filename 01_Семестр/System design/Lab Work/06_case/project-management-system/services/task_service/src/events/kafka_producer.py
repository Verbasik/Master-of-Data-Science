# project-management-system/services/task_service/src/events/kafka_producer.py

# Стандартные библиотеки
import json
import logging
from typing import Any, Dict, Optional
from uuid import UUID

# Сторонние библиотеки
from aiokafka import AIOKafkaProducer

# Настройка логирования
logger = logging.getLogger(__name__)


class KafkaTaskProducer:
    """
    Description:
        Класс для отправки событий задач в Kafka.
        Обеспечивает асинхронную отправку сообщений о создании задач.

    Args:
        bootstrap_servers: Адреса серверов Kafka
        topic: Название топика для отправки событий

    Attributes:
        bootstrap_servers (str): Адреса серверов Kafka
        topic (str): Название топика
        producer (Optional[AIOKafkaProducer]): Экземпляр Kafka producer

    Examples:
        >>> producer = KafkaTaskProducer("localhost:9092", "tasks")
        >>> await producer.start()
        >>> await producer.send_task_created_event({"id": "123"})
        >>> await producer.stop()
    """

    def __init__(self, bootstrap_servers: str, topic: str) -> None:
        self.bootstrap_servers: str = bootstrap_servers
        self.topic: str = topic
        self.producer: Optional[AIOKafkaProducer] = None

    async def start(self) -> None:
        """
        Description:
            Инициализирует и запускает Kafka producer.

        Raises:
            Exception: При ошибке запуска producer

        Examples:
            >>> await producer.start()
        """
        self.producer = AIOKafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8')
        )
        await self.producer.start()
        logger.info(f"Kafka producer started: {self.bootstrap_servers}")

    async def stop(self) -> None:
        """
        Description:
            Останавливает Kafka producer.

        Examples:
            >>> await producer.stop()
        """
        if self.producer:
            await self.producer.stop()
            logger.info("Kafka producer stopped")

    async def send_task_created_event(self, task_data: Dict[str, Any]) -> None:
        """
        Description:
            Отправляет событие о создании задачи в Kafka.

        Args:
            task_data: Данные созданной задачи

        Raises:
            RuntimeError: Если producer не запущен
            Exception: При ошибке отправки события

        Examples:
            >>> await producer.send_task_created_event({
            ...     "id": "123",
            ...     "title": "New Task"
            ... })
        """
        if not self.producer:
            raise RuntimeError("Producer not started")
        
        try:
            event = {
                "event_type": "task_created",
                "data": task_data
            }
            await self.producer.send_and_wait(self.topic, event)
            logger.info(f"Task created event sent: {task_data.get('id')}")
        except Exception as e:
            logger.error(f"Error sending task created event: {str(e)}")
            raise