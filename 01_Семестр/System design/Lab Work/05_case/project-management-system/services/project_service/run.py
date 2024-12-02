# project-management-system/services/project_service/run.py
import sys
import uvicorn
from pathlib import Path

# Добавление пути корневого каталога в sys.path для импорта модулей
root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

# Импорт приложения после добавления пути
from project_service.src.main import app

def run_server() -> None:
    """
    Description:
        Запускает сервер Uvicorn с приложением FastAPI.

    Args:
        None

    Returns:
        None

    Examples:
        >>> run_server()
        Сервер запускается и слушает входящие соединения на порту 8001.
    """
    uvicorn.run(
        "project_service.src.main:app",
        host="0.0.0.0",
        port=8001,
        reload=True
    )

if __name__ == "__main__":
    run_server()
