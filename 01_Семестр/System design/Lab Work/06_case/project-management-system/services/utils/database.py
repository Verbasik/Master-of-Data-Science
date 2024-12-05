# project-management-system/services/utils/database.py
import os
import logging
import asyncio
from dotenv import load_dotenv 
from typing import Optional, List, TypeVar, Type
from uuid import UUID
from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.declarative import declarative_base
from contextlib import contextmanager

# Загружаем переменные окружения из файла .env
load_dotenv()

# Создаем базовый класс для моделей
Base = declarative_base()

# Тип для моделей
ModelType = TypeVar("ModelType", bound=Base)

class PostgresManager:
    """
    Менеджер для работы с PostgreSQL базой данных.
    Обеспечивает единую точку доступа к БД для всех сервисов.
    """

    def __init__(
        self,
        database_url: str,
        max_pool_size: int = 20,
        min_pool_size: int = 5,
        max_reconnect_attempts: int = 5,
        reconnect_delay: int = 5
    ):
        """
        Инициализирует менеджер для подключения к базе данных.

        Args:
            database_url: URL базы данных PostgreSQL.
            max_pool_size: Максимальный размер пула соединений.
            min_pool_size: Минимальный размер пула соединений.
            max_reconnect_attempts: Максимальное количество попыток переподключения.
            reconnect_delay: Задержка между попытками переподключения в секундах.
        """
        self.logger = logging.getLogger(__name__)
        self.database_url = database_url
        self.engine = None
        self.SessionLocal = None
        self.max_pool_size = max_pool_size
        self.min_pool_size = min_pool_size
        self._reconnect_attempt = 0
        self._max_reconnect_attempts = max_reconnect_attempts
        self._reconnect_delay = reconnect_delay

    def connect(self) -> None:
        """
        Устанавливает соединение с PostgreSQL базой данных.
        
        Raises:
            SQLAlchemyError: Если не удалось подключиться к базе данных.
        """
        while self._reconnect_attempt < self._max_reconnect_attempts:
            try:
                if self.engine:
                    self.disconnect()
                self.engine = create_engine(
                    self.database_url,
                    pool_size=self.min_pool_size,
                    max_overflow=self.max_pool_size - self.min_pool_size,
                    pool_timeout=30,
                    pool_recycle=1800
                )
                # Создаем все таблицы
                Base.metadata.create_all(self.engine)

                self.SessionLocal = sessionmaker(
                    autocommit=False,
                    autoflush=False,
                    bind=self.engine
                )
                self.logger.info("Успешное подключение к PostgreSQL")
                self._reconnect_attempt = 0
                return
            except SQLAlchemyError as e:
                self._reconnect_attempt += 1
                if self._reconnect_attempt >= self._max_reconnect_attempts:
                    self.logger.error(f"Не удалось подключиться к PostgreSQL после {self._reconnect_attempt} попыток")
                    raise
                self.logger.warning(
                    f"Попытка подключения {self._reconnect_attempt} не удалась, "
                    f"повтор через {self._reconnect_delay} секунд"
                )
                asyncio.sleep(self._reconnect_delay)

    def disconnect(self) -> None:
        """Закрывает соединение с базой данных PostgreSQL."""
        if self.engine:
            self.engine.dispose()
            self.engine = None
            self.SessionLocal = None
            self.logger.info("Соединение с PostgreSQL закрыто")

    @contextmanager
    def get_db(self) -> Session:
        """
        Контекстный менеджер для работы с сессией базы данных.

        Yields:
            Session: Сессия базы данных.
        
        Raises:
            RuntimeError: Если база данных не подключена.
        """
        if not self.SessionLocal:
            raise RuntimeError("Database not connected")
        db = self.SessionLocal()
        try:
            yield db
        finally:
            db.close()

    def _handle_db_error(self, operation: str, error: Exception) -> None:
        """
        Логирует и обрабатывает ошибки, связанные с базой данных.

        Args:
            operation: Название операции, во время которой возникла ошибка.
            error: Исключение, возникшее во время операции.

        Raises:
            SQLAlchemyError: Если операция завершилась ошибкой.
        """
        self.logger.error(f"Database error during {operation}: {str(error)}")
        raise

    def create(self, model: Type[ModelType], obj_in: dict) -> ModelType:
        """
        Создает новый объект в базе данных.

        Args:
            model: Модель SQLAlchemy, в которую добавляется объект.
            obj_in: Данные для создания объекта.

        Returns:
            Созданный объект.
        """
        try:
            with self.get_db() as db:
                db_obj = model(**obj_in)
                db.add(db_obj)
                db.commit()
                db.refresh(db_obj)
                return db_obj
        except SQLAlchemyError as e:
            self._handle_db_error("create", e)

    def get(self, model: Type[ModelType], id: UUID) -> Optional[ModelType]:
        """
        Получает объект по его ID.

        Args:
            model: Модель SQLAlchemy.
            id: Идентификатор объекта.

        Returns:
            Объект, если он найден, иначе None.
        """
        try:
            with self.get_db() as db:
                return db.query(model).filter(model.id == id).first()
        except SQLAlchemyError as e:
            self._handle_db_error("get", e)

    def get_multi(
        self, 
        model: Type[ModelType], 
        *, 
        skip: int = 0, 
        limit: int = 100
    ) -> List[ModelType]:
        """
        Получает список объектов с поддержкой пагинации.

        Args:
            model: Модель SQLAlchemy.
            skip: Количество пропускаемых записей.
            limit: Максимальное количество записей.

        Returns:
            Список объектов модели.
        """
        try:
            with self.get_db() as db:
                return db.query(model).offset(skip).limit(limit).all()
        except SQLAlchemyError as e:
            self._handle_db_error("get_multi", e)

    def update(
        self, 
        model: Type[ModelType],
        id: UUID,
        obj_in: dict
    ) -> Optional[ModelType]:
        """
        Обновляет объект в базе данных.

        Args:
            model: Модель SQLAlchemy.
            id: Идентификатор объекта.
            obj_in: Данные для обновления.

        Returns:
            Обновленный объект, если он найден, иначе None.
        """
        try:
            with self.get_db() as db:
                db_obj = db.query(model).filter(model.id == id).first()
                if not db_obj:
                    return None
                for key, value in obj_in.items():
                    if hasattr(db_obj, key):
                        setattr(db_obj, key, value)
                db_obj.updated_at = datetime.utcnow()
                db.commit()
                db.refresh(db_obj)
                return db_obj
        except SQLAlchemyError as e:
            self._handle_db_error("update", e)

    def delete(self, model: Type[ModelType], id: UUID) -> bool:
        """
        Удаляет объект из базы данных.

        Args:
            model: Модель SQLAlchemy.
            id: Идентификатор объекта.

        Returns:
            True, если объект был успешно удален, иначе False.
        """
        try:
            with self.get_db() as db:
                obj = db.query(model).filter(model.id == id).first()
                if not obj:
                    return False
                db.delete(obj)
                db.commit()
                return True
        except SQLAlchemyError as e:
            self._handle_db_error("delete", e)

    def get_user_by_username(self, model: Type[ModelType], username: str) -> Optional[ModelType]:
        """
        Получает пользователя по имени пользователя.

        Args:
            model: Модель SQLAlchemy.
            username: Имя пользователя.

        Returns:
            Объект пользователя, если он найден, иначе None.
        """
        try:
            with self.get_db() as db:
                return db.query(model).filter(model.username == username).first()
        except SQLAlchemyError as e:
            self._handle_db_error("get_user_by_username", e)

    def get_user_by_email(self, model: Type[ModelType], email: str) -> Optional[ModelType]:
        """
        Получает пользователя по email.

        Args:
            model: Модель SQLAlchemy.
            email: Электронная почта пользователя.

        Returns:
            Объект пользователя, если он найден, иначе None.
        """
        try:
            with self.get_db() as db:
                return db.query(model).filter(model.email == email).first()
        except SQLAlchemyError as e:
            self._handle_db_error("get_user_by_email", e)

    def get_projects_by_owner(
        self, 
        model: Type[ModelType], 
        owner_id: UUID
    ) -> List[ModelType]:
        """
        Получает все проекты пользователя по его идентификатору.

        Args:
            model: Модель SQLAlchemy.
            owner_id: Идентификатор пользователя.

        Returns:
            Список проектов, принадлежащих пользователю.
        """
        try:
            with self.get_db() as db:
                return db.query(model).filter(model.owner_id == owner_id).all()
        except SQLAlchemyError as e:
            self._handle_db_error("get_projects_by_owner", e)

    def get_tasks_by_project(
        self, 
        model: Type[ModelType], 
        project_id: UUID
    ) -> List[ModelType]:
        """
        Получает все задачи проекта по его идентификатору.

        Args:
            model: Модель SQLAlchemy.
            project_id: Идентификатор проекта.

        Returns:
            Список задач проекта.
        """
        try:
            with self.get_db() as db:
                return db.query(model).filter(model.project_id == project_id).all()
        except SQLAlchemyError as e:
            self._handle_db_error("get_tasks_by_project", e)

    def get_tasks_by_assignee(
        self, 
        model: Type[ModelType], 
        assignee_id: UUID
    ) -> List[ModelType]:
        """
        Получает все задачи, назначенные определенному пользователю.

        Args:
            model: Модель SQLAlchemy.
            assignee_id: Идентификатор пользователя.

        Returns:
            Список задач, назначенных на пользователя.
        """
        try:
            with self.get_db() as db:
                return db.query(model).filter(model.assignee_id == assignee_id).all()
        except SQLAlchemyError as e:
            self._handle_db_error("get_tasks_by_assignee", e)

    def update_task_status(
        self, 
        model: Type[ModelType],
        task_id: UUID, 
        status: str
    ) -> Optional[ModelType]:
        """
        Обновляет статус задачи.

        Args:
            model: Модель SQLAlchemy.
            task_id: Идентификатор задачи.
            status: Новый статус задачи.

        Returns:
            Обновленный объект задачи, если он найден, иначе None.
        """
        return self.update(model, task_id, {"status": status})

# Создание экземпляра менеджера
db_manager = PostgresManager(
    database_url=os.getenv("DATABASE_URL")
)
