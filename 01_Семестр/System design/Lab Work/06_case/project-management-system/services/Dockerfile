# Stage 1: Base
FROM python:3.11-slim AS base

# Установка базовых зависимостей для Kafka
RUN pip install --no-cache-dir \
    six>=1.16.0 \
    kafka-python>=2.0.2

# Проверка установки Kafka-зависимостей
RUN python -c "from kafka.vendor.six.moves import range"

# Установка остальных зависимостей
RUN pip install --no-cache-dir \
    fastapi==0.109.1 \
    uvicorn==0.27.0 \
    psycopg2-binary==2.9.9 \
    python-jose[cryptography]==3.3.0 \
    passlib[bcrypt]==1.7.4 \
    python-multipart==0.0.7 \
    pydantic==2.7.1 \
    pydantic[email] \
    email-validator==2.1.0 \
    python-dotenv==1.0.0 \
    sqlalchemy==2.0.25 \
    alembic==1.13.1 \
    httpx==0.26.0 \
    pymongo==4.5.0 \
    motor==3.3.1 \
    pydantic-settings==2.1.0 \
    redis==5.0.1 \
    aiokafka==0.8.1 \
    prometheus-client==0.20.0 \
    python-json-logger==2.0.7

# Проверка всех зависимостей
RUN pip check

WORKDIR /app
COPY services/utils /app/utils

# Stage 2: User Service
FROM base AS user-service
COPY services/user_service /app/user_service
WORKDIR /app/user_service
ENV PYTHONPATH=/app
CMD ["python", "run.py"]

# Stage 3: Project Service
FROM base AS project-service
COPY services/project_service /app/project_service
WORKDIR /app/project_service
ENV PYTHONPATH=/app
CMD ["python", "run.py"]

# Stage 4: Task Service
FROM base AS task-service
COPY services/task_service /app/task_service
WORKDIR /app/task_service
ENV PYTHONPATH=/app
CMD ["python", "run.py"]

# Stage 5: Task Event Handler
FROM base AS task-event-handler
COPY services/task_service /app/task_service
WORKDIR /app/task_service
ENV PYTHONPATH=/app
CMD ["python", "-m", "src.events.event_handler"]