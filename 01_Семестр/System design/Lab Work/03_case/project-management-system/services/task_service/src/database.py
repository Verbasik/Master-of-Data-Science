# project-management-system/services/task_service/src/database.py
from utils.database import db_manager

# Реэкспорт db_manager для обратной совместимости
db = db_manager