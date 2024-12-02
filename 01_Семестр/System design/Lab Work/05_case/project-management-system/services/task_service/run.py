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