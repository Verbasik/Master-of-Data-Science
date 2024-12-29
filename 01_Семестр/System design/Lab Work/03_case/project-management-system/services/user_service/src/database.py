# project-management-system/services/user_service/src/database.py
# ============================
# БЛОК ИМПОРТОВ
# ============================
from utils.database import db_manager

# Реэкспорт db_manager для обратной совместимости
db = db_manager
