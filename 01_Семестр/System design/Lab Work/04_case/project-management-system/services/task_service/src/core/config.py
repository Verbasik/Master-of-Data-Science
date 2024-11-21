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

    @validator("MONGODB_URL")
    def validate_mongodb_url(cls, value: str) -> str:
        """
        Проверяет, что URL MongoDB имеет корректный формат.
        """
        if not value.startswith(("mongodb://", "mongodb+srv://")):
            raise ValueError("Invalid MongoDB URL format")
        return value

    @validator("USER_SERVICE_URL", "PROJECT_SERVICE_URL", "TASK_SERVICE_URL")
    def validate_service_urls(cls, value: str) -> str:
        """
        Проверяет, что URL других сервисов имеет корректный формат.
        """
        if not value.startswith(("http://", "https://")):
            raise ValueError("Invalid service URL format")
        return value

    class Config:
        """
        Дополнительные настройки Pydantic.
        """
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
