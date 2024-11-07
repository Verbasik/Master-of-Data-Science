# project-management-system/services/project_service/src/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api import routes
from .database import db

app = FastAPI(title="Project Service", description="API для управления проектами")

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Подключение роутов
app.include_router(routes.router)

@app.on_event("startup")
async def startup_event():
    """
    Description:
      Функция, выполняемая при запуске приложения.
      Здесь можно добавить инициализацию, если необходимо.
    """
    print("Project Service started")
    try:
        # Инициализируем подключение к БД
        db.connect()
        print("Project Service started - Database connected successfully")
    except Exception as e:
        print(f"Error connecting to database: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """
    Description:
      Функция, выполняемая при остановке приложения.
    """
    try:
        # Закрываем подключение к БД
        db.disconnect()
        print("Database connection closed")
    except Exception as e:
        print(f"Error disconnecting from database: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)