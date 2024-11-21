# project-management-system/services/project_service/src/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api import routes

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
    Описание:
      Функция, выполняемая при запуске приложения.
      Здесь можно добавить инициализацию, если необходимо.
    """
    print("Project Service started")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)