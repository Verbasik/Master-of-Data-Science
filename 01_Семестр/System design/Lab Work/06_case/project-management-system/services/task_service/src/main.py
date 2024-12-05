# project-management-system/services/task_service/src/main.py

# Стандартные библиотеки
import json
import logging
from typing import Dict, Any

# Сторонние библиотеки
from aiokafka import AIOKafkaConsumer
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient

# Локальные импорты
from .api import routes
from .core.config import settings
from .database import db
from .events.kafka_producer import KafkaTaskProducer

# Настройка логирования
logger = logging.getLogger(__name__)

# Инициализация FastAPI
app = FastAPI(
    title="Task Service",
    description="API для управления задачами"
)

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Подключение маршрутов
app.include_router(routes.router)


class TaskEventHandler:
    """
    Description:
        Обработчик событий задач из Kafka.

    Args:
        bootstrap_servers: Адреса серверов Kafka
        topic: Название топика
        mongodb_url: URL подключения к MongoDB

    Examples:
        >>> handler = TaskEventHandler("localhost:9092", "tasks", "mongodb://localhost")
        >>> await handler.start()
    """

    def __init__(
        self,
        bootstrap_servers: str,
        topic: str,
        mongodb_url: str
    ) -> None:
        self.bootstrap_servers: str = bootstrap_servers
        self.topic: str = topic
        self.mongodb_url: str = mongodb_url
        self.consumer: Optional[AIOKafkaConsumer] = None
        self.db: Optional[AsyncIOMotorClient] = None

    async def start(self) -> None:
        """
        Description:
            Запускает обработчик событий.

        Raises:
            Exception: При ошибке запуска обработчика
        """
        client = AsyncIOMotorClient(self.mongodb_url)
        self.db = client.task_service.tasks

        self.consumer = AIOKafkaConsumer(
            self.topic,
            bootstrap_servers=self.bootstrap_servers,
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )
        await self.consumer.start()
        logger.info("Task event handler started")

        try:
            async for msg in self.consumer:
                await self.handle_event(msg.value)
        finally:
            await self.consumer.stop()

    async def handle_event(self, event: Dict[str, Any]) -> None:
        """
        Description:
            Обрабатывает полученное событие.

        Args:
            event: Данные события
        """
        try:
            if event["event_type"] == "task_created":
                await self.handle_task_created(event["data"])
        except Exception as e:
            logger.error(f"Error handling event: {str(e)}")

    async def handle_task_created(self, task_data: Dict[str, Any]) -> None:
        """
        Description:
            Обрабатывает событие создания задачи.

        Args:
            task_data: Данные созданной задачи
        """
        try:
            await self.db.insert_one(task_data)
            logger.info(f"Task created in database: {task_data.get('id')}")
        except Exception as e:
            logger.error(f"Error saving task to database: {str(e)}")


@app.on_event("startup")
async def startup_event() -> None:
    """
    Description:
        Инициализирует сервисы при запуске приложения.

    Raises:
        Exception: При ошибке инициализации
    """
    logger.info("Task Service starting...")
    try:
        db.connect()
        logger.info("Database connected successfully")

        app.state.kafka_producer = KafkaTaskProducer(
            bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
            topic=settings.KAFKA_TASK_TOPIC
        )
        await app.state.kafka_producer.start()
        logger.info("KafkaTaskProducer initialized successfully")

    except Exception as e:
        logger.error(f"Startup error: {str(e)}")
        raise


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """
    Description:
        Освобождает ресурсы при остановке приложения.
    """
    try:
        db.disconnect()
        logger.info("Database disconnected")
        await app.state.kafka_producer.stop()
        logger.info("KafkaTaskProducer stopped")
    except Exception as e:
        logger.error(f"Shutdown error: {str(e)}")


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """
    Description:
        Проверяет состояние сервиса.

    Returns:
        Dict[str, str]: Статус сервиса
    """
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)