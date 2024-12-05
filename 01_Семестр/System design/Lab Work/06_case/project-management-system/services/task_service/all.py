# project-management-system/services/task_service/run.py
import sys
from pathlib import Path

# Добавляем путь к корню проекта
root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

import uvicorn
from task_service.src.main import app

if __name__ == "__main__":
    uvicorn.run(
        "task_service.src.main:app",
        host="0.0.0.0",
        port=8002,
        reload=True
    )

# project-management-system/services/task_service/src/main.py

# --- Импорты сторонних библиотек ---
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# --- Импорты модулей проекта ---
from .api import routes
from .database import db

import json
import logging
from typing import Dict, Any
from aiokafka import AIOKafkaConsumer
from motor.motor_asyncio import AsyncIOMotorClient

# Инициализация FastAPI приложения
app = FastAPI(title="Task Service", description="API для управления задачами")

logger = logging.getLogger(__name__)

# --- Настройка CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Подключение маршрутов ---
app.include_router(routes.router)

class TaskEventHandler:
    def __init__(self, bootstrap_servers: str, topic: str, mongodb_url: str):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.mongodb_url = mongodb_url
        self.consumer = None
        self.db = None

    async def start(self):
        # Инициализация MongoDB
        client = AsyncIOMotorClient(self.mongodb_url)
        self.db = client.task_service.tasks

        # Инициализация Kafka Consumer
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

    async def handle_event(self, event: Dict[str, Any]):
        try:
            if event["event_type"] == "task_created":
                await self.handle_task_created(event["data"])
        except Exception as e:
            logger.error(f"Error handling event: {str(e)}")

    async def handle_task_created(self, task_data: Dict[str, Any]):
        try:
            await self.db.insert_one(task_data)
            logger.info(f"Task created in database: {task_data.get('id')}")
        except Exception as e:
            logger.error(f"Error saving task to database: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """
    Description:
        Функция, выполняемая при запуске приложения. Инициализирует подключение к базе данных.
    """
    print("Task Service started")
    try:
        # Инициализируем подключение к БД
        db.connect()
        print("Task Service - Database connected successfully")

        # Инициализация Kafka producer
        app.state.kafka_producer = KafkaTaskProducer(
            bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
            topic=settings.KAFKA_TASK_TOPIC
        )
        await app.state.kafka_producer.start()
        print("Task Service - KafkaTaskProducer connected successfully")

    except Exception as e:
        print(f"Error connecting to database: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """
    Description:
        Функция, выполняемая при остановке приложения. Закрывает подключение к базе данных.
    """
    try:
        # Закрываем подключение к БД
        db.disconnect()
        print("Database connection closed")
        await app.state.kafka_producer.stop()
        print("Kafka_producer connection closed")
    except Exception as e:
        print(f"Error disconnecting from database: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)

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
    def __init__(self):
        self.client: Optional[AsyncIOMotorClient] = None
        self.db = None
        self.tasks = None

    async def connect(self) -> None:
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
        if self.client:
            self.client.close()
            logger.info("Соединение с MongoDB закрыто.")

    async def ensure_indexes(self) -> None:
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
        try:
            task_dict = await self.tasks.find_one({"_id": str(task_id)})
            if task_dict:
                return MongoTask(**self._convert_ids_to_uuid(task_dict))
            return None
        except Exception as e:
            logger.error(f"Ошибка получения задачи: {e}")
            raise

    async def update_task(self, task_id: UUID, update_data: Dict[str, Any]) -> Optional[MongoTask]:
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
        try:
            result = await self.tasks.delete_one({"_id": str(task_id)})
            return result.deleted_count > 0
        except Exception as e:
            logger.error(f"Ошибка удаления задачи: {e}")
            raise

    def _convert_uuids_to_str(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            key: (str(value) if isinstance(value, UUID) else value)
            for key, value in data.items()
        }

    def _convert_ids_to_uuid(self, data: Dict[str, Any]) -> Dict[str, Any]:
        for field in ["_id", "project_id", "creator_id", "assignee_id"]:
            if field in data and data[field]:
                data[field] = UUID(data[field])
        return data
    
# Реэкспорт db_manager для обратной совместимости
db = db_manager

# project-management-system/services/task_service/src/auth.py

# --- Импорты стандартных библиотек ---
import os
from datetime import datetime, timedelta

# --- Импорты сторонних библиотек ---
from dotenv import load_dotenv
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from pydantic import BaseModel
from uuid import UUID
import httpx

# Загрузка переменных окружения
load_dotenv()

# --- Настройки JWT ---
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM  = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
USER_SERVICE_URL = os.getenv("USER_SERVICE_URL")

# --- Настройка OAuth2 ---
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class TokenData(BaseModel):
    username: Optional[str] = None

class User(BaseModel):
    id: UUID
    username: str
    email: str
    is_active: bool

async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: Optional[str] = payload.get("sub")
        if username is None:
            raise credentials_exception

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{USER_SERVICE_URL}/users/me",
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=10.0
                )

                if response.status_code == 200:
                    user_data = response.json()
                    return User(**user_data)
                else:
                    raise credentials_exception
            except httpx.RequestError as e:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"User service unavailable: {str(e)}"
                )
    except JWTError:
        raise credentials_exception

# project-management-system/services/task_service/src/services/command_service.py
from typing import Optional
from uuid import UUID
import logging
from ..models.api_models import TaskCreate, TaskUpdate
from ..events.kafka_producer import KafkaTaskProducer

logger = logging.getLogger(__name__)

class TaskCommandService:
    def __init__(self, kafka_producer: KafkaTaskProducer):
        self.kafka_producer = kafka_producer

    async def create_task(self, task_data: TaskCreate, creator_id: UUID) -> Dict[str, Any]:
        try:
            task_dict = task_data.dict()
            task_dict["creator_id"] = creator_id
            task_dict["id"] = str(UUID.uuid4())
            
            # Отправляем событие в Kafka
            await self.kafka_producer.send_task_created_event(task_dict)
            
            return task_dict
        except Exception as e:
            logger.error(f"Error in create_task command: {str(e)}")
            raise

# project-management-system/services/task_service/src/services/query_service.py
from typing import List, Optional, Dict, Any
from uuid import UUID
import logging
from ..models.mongo_models import MongoTask
from ..database import MongoDBManager

logger = logging.getLogger(__name__)

class TaskQueryService:
    def __init__(self, db_manager: MongoDBManager):
        self.db = db_manager

    async def get_task(self, task_id: UUID) -> Optional[MongoTask]:
        try:
            return await self.db.get_task(task_id)
        except Exception as e:
            logger.error(f"Error in get_task query: {str(e)}")
            raise

    async def get_tasks_by_criteria(self, criteria: Dict[str, Any]) -> List[MongoTask]:
        try:
            return await self.db.get_tasks_by_criteria(criteria)
        except Exception as e:
            logger.error(f"Error in get_tasks_by_criteria query: {str(e)}")
            raise

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
    
    def __init__(self, db_manager):
        self.db = db_manager

    async def create_task(self, task_data: TaskCreate, creator_id: UUID) -> MongoTask:
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
        try:
            update_data = task_update.dict(exclude_unset=True)
            updated_task = await self.db.update_task(task_id, update_data)
            logger.info(f"Задача {task_id} успешно обновлена.")
            return updated_task
        except Exception as e:
            logger.error(f"Ошибка обновления задачи: {str(e)}")
            raise

    async def delete_task(self, task_id: UUID) -> bool:
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
        try:
            tasks = await self.db.get_tasks_by_project(project_id)
            logger.info(f"Найдено {len(tasks)} задач для проекта {project_id}.")
            return tasks
        except Exception as e:
            logger.error(f"Ошибка получения задач проекта: {str(e)}")
            raise

    async def get_tasks_by_criteria(self, criteria: Dict[str, Any]) -> List[MongoTask]:
        try:
            tasks = await self.db.get_tasks_by_criteria(criteria)
            logger.info(f"Найдено {len(tasks)} задач по критериям {criteria}.")
            return tasks
        except Exception as e:
            logger.error(f"Ошибка получения задач по критериям: {str(e)}")
            raise

# project-management-system/services/task_service/src/models/api_models.py
from typing import Optional
from uuid import UUID, uuid4
from datetime import datetime
from pydantic import BaseModel, Field
from .database_models import TaskStatus, TaskPriority

class TaskBase(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    description: str = Field(..., max_length=2000)
    priority: TaskPriority = Field(default=TaskPriority.MEDIUM)

    class Config:
        from_attributes = True

class TaskCreate(TaskBase):
    project_id: UUID
    assignee_id: Optional[UUID] = None

class Task(TaskBase):
    id: UUID = Field(default_factory=uuid4)
    status: TaskStatus = Field(default=TaskStatus.CREATED)
    project_id: UUID
    creator_id: UUID
    assignee_id: Optional[UUID]
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        rom_attributes = True

class TaskUpdate(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=2000)
    status: Optional[TaskStatus] = None
    priority: Optional[TaskPriority] = None
    assignee_id: Optional[UUID] = None

class TaskResponse(Task):

    @classmethod
    def from_mongo(cls, mongo_task: "MongoTask") -> "TaskResponse":
        return cls(
            id          = mongo_task.id,
            title       = mongo_task.title,
            description = mongo_task.description,
            status      = mongo_task.status,
            priority    = mongo_task.priority,
            project_id  = mongo_task.project_id,
            creator_id  = mongo_task.creator_id,
            assignee_id = mongo_task.assignee_id,
            created_at  = mongo_task.created_at,
            updated_at  = mongo_task.updated_at,
        )

    class Config:
        from_attributes = True

# project-management-system/services/task_service/src/models/database_models.py
import uuid
import enum
from utils.database import Base

from sqlalchemy import Column, String, Text, DateTime, ForeignKey, Enum as SQLAlchemyEnum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

class TaskStatus(str, enum.Enum):
    CREATED     = "created"
    IN_PROGRESS = "in_progress"
    ON_REVIEW   = "on_review"
    COMPLETED   = "completed"

class TaskPriority(str, enum.Enum):
    LOW    = "low"
    MEDIUM = "medium"
    HIGH   = "high"

class Task(Base):
    
    __tablename__ = "tasks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(200), nullable=False)
    description = Column(Text)
    status = Column(SQLAlchemyEnum(TaskStatus), default=TaskStatus.CREATED, nullable=False)
    priority = Column(SQLAlchemyEnum(TaskPriority), default=TaskPriority.MEDIUM, nullable=False)
    
    project_id = Column(UUID(as_uuid=True), nullable=False)
    creator_id = Column(UUID(as_uuid=True), nullable=False)
    assignee_id = Column(UUID(as_uuid=True), nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    def __repr__(self) -> str:
        return f"<Task(id={self.id}, title={self.title})>"

# project-management-system/services/task_service/src/models/mongo_models.py
from datetime import datetime
from typing import Optional, List, Dict
from uuid import UUID, uuid4
from pydantic import BaseModel, Field
from ..models.database_models import TaskStatus, TaskPriority

class MongoTask(BaseModel):

    id: UUID = Field(default_factory=uuid4, alias="_id", description="Уникальный идентификатор задачи")
    title: str = Field(..., min_length=1, max_length=200, description="Название задачи (от 1 до 200 символов)")
    description: str = Field(..., max_length=2000, description="Описание задачи (до 2000 символов)")
    status: TaskStatus = Field(default=TaskStatus.CREATED, description="Статус задачи")
    priority: TaskPriority = Field(default=TaskPriority.MEDIUM, description="Приоритет задачи")
    project_id: UUID = Field(..., description="ID проекта, связанного с задачей")
    creator_id: UUID = Field(..., description="ID создателя задачи")
    assignee_id: Optional[UUID] = Field(None, description="ID исполнителя задачи")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Дата создания задачи")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Дата последнего обновления задачи")

    metadata: Dict[str, str] = Field(default_factory=dict, description="Произвольные метаданные задачи")
    tags: List[str] = Field(default_factory=list, description="Список тегов задачи")

    class Config:
        populate_by_name = True
        json_encoders = {
            UUID: str,                            # Преобразование UUID в строку для сериализации
            datetime: lambda dt: dt.isoformat(),  # Преобразование даты в ISO формат
        }

    @classmethod
    def from_db(cls, data: Dict) -> Optional["MongoTask"]:
        if not data:
            return None

        # Конвертируем строковые ID в UUID, если необходимо
        for field in ["_id", "project_id", "creator_id", "assignee_id"]:
            if field in data and data[field]:
                try:
                    data[field] = UUID(data[field])
                except ValueError as e:
                    raise ValueError(f"Invalid UUID for field '{field}': {data[field]}") from e

        return cls(**data)
    
# project-management-system/services/task_service/src/events/kafka_producer.py
from aiokafka import AIOKafkaProducer
import json
import logging
from typing import Any, Dict
from uuid import UUID

logger = logging.getLogger(__name__)

class KafkaTaskProducer:
    def __init__(self, bootstrap_servers: str, topic: str):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.producer = None

    async def start(self):
        self.producer = AIOKafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8')
        )
        await self.producer.start()
        logger.info(f"Kafka producer started: {self.bootstrap_servers}")

    async def stop(self):
        if self.producer:
            await self.producer.stop()
            logger.info("Kafka producer stopped")

    async def send_task_created_event(self, task_data: Dict[str, Any]):
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

# project-management-system/services/task_service/src/core/config.py

# Импорты библиотек
from pydantic import Field, validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Настройки приложения, загружаемые из переменных окружения или .env файла.
    """

    # Настройки MongoDB
    MONGODB_URL: str     = Field(default="mongodb://mongodb:27017", description="URL для подключения к MongoDB")
    MONGODB_DB_NAME: str = Field(default="task_service", description="Имя базы данных MongoDB")

    # URL других сервисов
    USER_SERVICE_URL: str    = Field(default="http://localhost:8000", description="URL пользовательского сервиса")
    PROJECT_SERVICE_URL: str = Field(default="http://localhost:8001", description="URL сервиса проектов")
    TASK_SERVICE_URL: str    = Field(default="http://localhost:8002", description="URL сервиса задач")

    # Настройки базы данных
    DATABASE_URL: str = Field(..., description="URL базы данных")

    # Настройки безопасности
    SECRET_KEY: str = Field(..., description="Секретный ключ приложения")

    # Kafka settings
    KAFKA_BOOTSTRAP_SERVERS: str
    KAFKA_TASK_TOPIC: str
    KAFKA_CONSUMER_GROUP: str
    KAFKA_MAX_RETRIES: int = 3
    KAFKA_RETRY_DELAY: int = 1

    @validator("MONGODB_URL")
    def validate_mongodb_url(cls, value: str) -> str:
        if not value.startswith(("mongodb://", "mongodb+srv://")):
            raise ValueError("Invalid MongoDB URL format")
        return value

    @validator("USER_SERVICE_URL", "PROJECT_SERVICE_URL", "TASK_SERVICE_URL")
    def validate_service_urls(cls, value: str) -> str:
        if not value.startswith(("http://", "https://")):
            raise ValueError("Invalid service URL format")
        return value

    class Config:
        env_file = ".env"
        case_sensitive = False   # Игнорировать регистр в именах переменных
        populate_by_name = True  # Поддержка заполнения через Field(name="...")

# Создаем экземпляр настроек
settings = Settings()

# Экспортируем значения для удобства импорта в других частях приложения
MONGODB_URL         = settings.MONGODB_URL
MONGODB_DB_NAME     = settings.MONGODB_DB_NAME
USER_SERVICE_URL    = settings.USER_SERVICE_URL
PROJECT_SERVICE_URL = settings.PROJECT_SERVICE_URL
TASK_SERVICE_URL    = settings.TASK_SERVICE_URL

# project-management-system/services/task_service/src/api/routes.py

# Импорты библиотек
from typing import List, Optional, AsyncGenerator
from uuid import UUID

# Импорты FastAPI
from fastapi import APIRouter, Depends, HTTPException, Query, status

# Логирование
import logging

# Локальные импорты
from ..services.command_service import TaskCommandService
from ..services.query_service import TaskQueryService
from ..services.task_service import TaskService
from ..models.api_models import TaskCreate, TaskUpdate, TaskResponse
from ..models.mongo_models import TaskStatus, TaskPriority
from ..auth import get_current_user, User
from ..database import MongoDBManager

# Логирование
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Маршрутизатор FastAPI
router = APIRouter(prefix="/tasks", tags=["tasks"])


# Dependency Injection функции
async def get_task_command_service() -> AsyncGenerator[TaskCommandService, None]:
    kafka_producer = None
    try:
        # Инициализация Kafka producer
        kafka_producer = KafkaTaskProducer(
            bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
            topic=settings.KAFKA_TASK_TOPIC
        )
        await kafka_producer.start()
        
        # Создание command service
        command_service = TaskCommandService(kafka_producer)
        yield command_service
    except Exception as e:
        logger.error(f"Failed to initialize command service: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to initialize command service"
        )
    finally:
        if kafka_producer:
            await kafka_producer.stop()

async def get_task_query_service() -> AsyncGenerator[TaskQueryService, None]:
    db_manager = None
    try:
        # Инициализация MongoDB manager
        db_manager = MongoDBManager()
        await db_manager.connect()
        
        # Создание query service
        query_service = TaskQueryService(db_manager)
        yield query_service
    except Exception as e:
        logger.error(f"Failed to initialize query service: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to initialize query service"
        )
    finally:
        if db_manager:
            await db_manager.disconnect()

# Dependency для получения экземпляра TaskService
async def get_task_service() -> AsyncGenerator[TaskService, None]:
    db_manager = None
    try:
        # Создаем экземпляр и сохраняем его
        db_manager = MongoDBManager()
        # Подключаемся и сохраняем результат
        connected_manager = await db_manager.connect()
        logger.info("Подключение к MongoDB успешно")

        service = TaskService(connected_manager)
        logger.info("TaskService успешно инициализирован")

        yield service

    except Exception as e:
        logger.error(f"Не удалось инициализировать TaskService: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ошибка инициализации сервиса задач."
        )

    finally:
        if db_manager:
            logger.info("Закрытие подключения к MongoDB")
            await db_manager.disconnect()
            logger.info("Подключение к MongoDB закрыто")

@router.post("", response_model=TaskResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_task(
    task: TaskCreate,
    current_user: User = Depends(get_current_user),
    command_service: TaskCommandService = Depends(get_task_command_service)
) -> TaskResponse:
    try:
        task_dict = await command_service.create_task(task, current_user.id)
        return TaskResponse(**task_dict)
    except Exception as e:
        logger.error(f"Error creating task: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating task: {str(e)}"
        )

@router.get("", response_model=List[TaskResponse])
async def get_tasks(
    project_id: Optional[UUID] = None,
    assignee_id: Optional[UUID] = None,
    status: Optional[TaskStatus] = None,
    priority: Optional[TaskPriority] = None,
    current_user: User = Depends(get_current_user),
    query_service: TaskQueryService = Depends(get_task_query_service)
) -> List[TaskResponse]:
    try:
        criteria = {
            k: v for k, v in {
                "project_id": project_id,
                "assignee_id": assignee_id,
                "status": status,
                "priority": priority
            }.items() if v is not None
        }
        tasks = await query_service.get_tasks_by_criteria(criteria)
        return [TaskResponse.from_mongo(task) for task in tasks]
    except Exception as e:
        logger.error(f"Error fetching tasks: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching tasks: {str(e)}"
        )


@router.get("/{task_id}", response_model=TaskResponse)
async def get_task(
    task_id: UUID,
    current_user: User = Depends(get_current_user),
    task_service: TaskService = Depends(get_task_service)
) -> TaskResponse:
    try:
        task = await task_service.get_task(task_id)
        if task is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task {task_id} not found"
            )
        return TaskResponse.from_mongo(task)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching task {task_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching task: {str(e)}"
        )

@router.put("/{task_id}", response_model=TaskResponse)
async def update_task(
    task_id: UUID,
    task_update: TaskUpdate,
    current_user: User = Depends(get_current_user),
    task_service: TaskService = Depends(get_task_service)
) -> TaskResponse:
    try:
        # Проверяем существование задачи
        existing_task = await task_service.get_task(task_id)
        if existing_task is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task {task_id} not found"
            )

        # Проверяем права доступа
        if (existing_task.creator_id != current_user.id and 
            existing_task.assignee_id != current_user.id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to update this task"
            )

        logger.info(f"Updating task {task_id} by user {current_user.id}")
        updated_task = await task_service.update_task(task_id, task_update)
        if updated_task is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update task"
            )
        logger.info(f"Task {task_id} updated successfully")
        return TaskResponse.from_mongo(updated_task)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating task {task_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating task: {str(e)}"
        )

@router.delete("/{task_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_task(
    task_id: UUID,
    current_user: User = Depends(get_current_user),
    task_service: TaskService = Depends(get_task_service)
):
    try:
        # Проверяем существование задачи
        existing_task = await task_service.get_task(task_id)
        if existing_task is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task {task_id} not found"
            )

        # Проверяем права доступа
        if existing_task.creator_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to delete this task"
            )

        logger.info(f"Deleting task {task_id} by user {current_user.id}")
        success = await task_service.delete_task(task_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete task"
            )
        logger.info(f"Task {task_id} deleted successfully")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting task {task_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting task: {str(e)}"
        )

@router.patch("/{task_id}/status", response_model=TaskResponse)
async def update_task_status(
    task_id: UUID,
    status: TaskStatus,
    current_user: User = Depends(get_current_user),
    task_service: TaskService = Depends(get_task_service)
) -> TaskResponse:
    try:
        # Проверяем существование задачи
        existing_task = await task_service.get_task(task_id)
        if existing_task is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task {task_id} not found"
            )

        # Проверяем права доступа
        if (existing_task.creator_id != current_user.id and 
            existing_task.assignee_id != current_user.id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to update this task status"
            )

        logger.info(f"Updating status of task {task_id} to {status} by user {current_user.id}")
        task_update = TaskUpdate(status=status)
        updated_task = await task_service.update_task(task_id, task_update)
        if updated_task is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update task status"
            )
        logger.info(f"Task {task_id} status updated successfully")
        return TaskResponse.from_mongo(updated_task)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating task status {task_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating task status: {str(e)}"
        )

@router.patch("/{task_id}/assignee", response_model=TaskResponse)
async def update_task_assignee(
    task_id: UUID,
    assignee_id: Optional[UUID],
    current_user: User = Depends(get_current_user),
    task_service: TaskService = Depends(get_task_service)
) -> TaskResponse:
    try:
        # Проверяем существование задачи
        existing_task = await task_service.get_task(task_id)
        if existing_task is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task {task_id} not found"
            )

        # Проверяем права доступа
        if existing_task.creator_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to update task assignee"
            )

        logger.info(f"Updating assignee of task {task_id} to {assignee_id} by user {current_user.id}")
        task_update = TaskUpdate(assignee_id=assignee_id)
        updated_task = await task_service.update_task(task_id, task_update)
        if updated_task is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update task assignee"
            )
        logger.info(f"Task {task_id} assignee updated successfully")
        return TaskResponse.from_mongo(updated_task)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating task assignee {task_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating task assignee: {str(e)}"
        )