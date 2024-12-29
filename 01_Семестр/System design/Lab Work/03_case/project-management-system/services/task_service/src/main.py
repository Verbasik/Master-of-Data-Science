# project-management-system/services/task_service/src/main.py

# --- Импорты сторонних библиотек ---
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# --- Импорты модулей проекта ---
from .api import routes
from .database import db

# Инициализация FastAPI приложения
app = FastAPI(title="Task Service", description="API для управления задачами")

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
    except Exception as e:
        print(f"Error disconnecting from database: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
