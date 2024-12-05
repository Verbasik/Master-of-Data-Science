# project-management-system/services/task_service/src/core/config.py

# Standard library imports
import os
from typing import Any

# Third-party imports
from pydantic import Field, validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Description:
        Класс настроек приложения, управляющий конфигурацией сервиса задач.
        Загружает настройки из переменных окружения или .env файла.

    Args:
        BaseSettings: Базовый класс настроек от pydantic

    Attributes:
        MONGODB_URL (str): URL подключения к MongoDB
        MONGODB_DB_NAME (str): Название базы данных MongoDB
        USER_SERVICE_URL (str): URL сервиса пользователей
        PROJECT_SERVICE_URL (str): URL сервиса проектов
        TASK_SERVICE_URL (str): URL сервиса задач
        DATABASE_URL (str): URL основной базы данных
        SECRET_KEY (str): Секретный ключ для безопасности
        KAFKA_*: Настройки для Kafka

    Raises:
        ValueError: Если формат URL некорректен

    Examples:
        >>> settings = Settings()
        >>> print(settings.MONGODB_URL)
        'mongodb://mongodb:27017'
    """

    # MongoDB settings
    MONGODB_URL: str = Field(
        default="mongodb://mongodb:27017",
        description="URL для подключения к MongoDB"
    )
    MONGODB_DB_NAME: str = Field(
        default="task_service",
        description="Имя базы данных MongoDB"
    )

    # Microservice URLs
    USER_SERVICE_URL: str = Field(
        default="http://localhost:8000",
        description="URL пользовательского сервиса"
    )
    PROJECT_SERVICE_URL: str = Field(
        default="http://localhost:8001",
        description="URL сервиса проектов"
    )
    TASK_SERVICE_URL: str = Field(
        default="http://localhost:8002",
        description="URL сервиса задач"
    )

    # Database settings
    DATABASE_URL: str = Field(
        ...,
        description="URL базы данных"
    )

    # Security settings
    SECRET_KEY: str = Field(
        ...,
        description="Секретный ключ приложения"
    )

    # Kafka settings
    KAFKA_BOOTSTRAP_SERVERS: str
    KAFKA_TASK_TOPIC: str
    KAFKA_CONSUMER_GROUP: str
    KAFKA_MAX_RETRIES: int = 3
    KAFKA_RETRY_DELAY: int = 1

    @validator("MONGODB_URL")
    def validate_mongodb_url(cls, value: str) -> str:
        """
        Description:
            Проверяет корректность формата URL для MongoDB.

        Args:
            value: URL для проверки

        Returns:
            str: Проверенный URL

        Raises:
            ValueError: Если URL имеет некорректный формат

        Examples:
            >>> Settings.validate_mongodb_url("mongodb://localhost:27017")
            'mongodb://localhost:27017'
        """
        if not value.startswith(("mongodb://", "mongodb+srv://")):
            raise ValueError("Invalid MongoDB URL format")
        return value

    @validator("USER_SERVICE_URL", "PROJECT_SERVICE_URL", "TASK_SERVICE_URL")
    def validate_service_urls(cls, value: str) -> str:
        """
        Description:
            Проверяет корректность формата URL для микросервисов.

        Args:
            value: URL для проверки

        Returns:
            str: Проверенный URL

        Raises:
            ValueError: Если URL имеет некорректный формат

        Examples:
            >>> Settings.validate_service_urls("http://localhost:8000")
            'http://localhost:8000'
        """
        if not value.startswith(("http://", "https://")):
            raise ValueError("Invalid service URL format")
        return value

    class Config:
        """
        Description:
            Дополнительные настройки для класса Settings.
        
        Attributes:
            env_file (str): Путь к файлу с переменными окружения
            case_sensitive (bool): Флаг чувствительности к регистру
            populate_by_name (bool): Флаг заполнения по имени
        """
        env_file = ".env"
        case_sensitive = False
        populate_by_name = True


# Создаем единственный экземпляр настроек
settings = Settings()

# Экспортируем настройки для удобства импорта
MONGODB_URL         = settings.MONGODB_URL
MONGODB_DB_NAME     = settings.MONGODB_DB_NAME
USER_SERVICE_URL    = settings.USER_SERVICE_URL
PROJECT_SERVICE_URL = settings.PROJECT_SERVICE_URL
TASK_SERVICE_URL    = settings.TASK_SERVICE_URL