# project-management-system/services/user_service/src/main.py
# ============================
# БЛОК ИМПОРТОВ
# ============================
# Импорты из сторонних библиотек
# Эти модули предоставляют основу для разработки FastAPI-приложений, включая 
# создание приложения, настройку middleware и авторизацию.
from fastapi import FastAPI                         # Основной класс для создания приложения FastAPI
from fastapi.middleware.cors import CORSMiddleware  # Middleware для поддержки CORS
from fastapi.security import OAuth2PasswordBearer   # Токен-бэрер для авторизации OAuth2

# Импорты внутренних модулей проекта
# Эти модули реализуют бизнес-логику, взаимодействие с БД и маршрутизацию.
from api import routes              # Маршруты API-приложения
from database import db             # Модуль для работы с базой данных
from models import UserInDB         # Модель пользователя, хранящаяся в БД
from auth import get_password_hash  # Функция для хеширования пароля


# Настройка схемы авторизации OAuth2
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Создание экземпляра приложения FastAPI с заданным названием и описанием
app = FastAPI(title="User Service", description="API для управления пользователями")

# Настройка CORS (Cross-Origin Resource Sharing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # В продакшене следует указать конкретные домены для безопасности
    allow_credentials=True,  # Разрешение на передачу учетных данных
    allow_methods=["*"],     # Разрешение всех HTTP-методов (GET, POST и т.д.)
    allow_headers=["*"],     # Разрешение всех заголовков
)

# Подключение роутов из внешнего модуля
app.include_router(routes.router)

@app.on_event("startup")
async def startup_event():
    """
    Описание:
      Функция, выполняемая при запуске приложения.
      Создает мастер-пользователя, если он еще не существует.
    """
    # Проверка, существует ли уже мастер-пользователь с именем 'admin'
    if not db.get_user_by_username("admin"):
        # Создание нового пользователя с заданными параметрами
        admin_user = UserInDB(
            username="admin",                             # Имя пользователя
            email="admin@example.com",                    # Электронная почта администратора
            hashed_password=get_password_hash("secret"),  # Хешированный пароль
            is_active=True                                # Статус активности пользователя
        )
        # Сохранение нового администратора в базе данных
        db.create_user(admin_user)
        print("Мастер-пользователь создан.")

# Запуск приложения, если этот файл является главным
if __name__ == "__main__":
    import uvicorn
    # Запуск сервера с указанными параметрами
    uvicorn.run(app, host="0.0.0.0", port=8000)
