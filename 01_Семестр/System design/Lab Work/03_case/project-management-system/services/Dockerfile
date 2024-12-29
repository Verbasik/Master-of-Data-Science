# services/Dockerfile
# Stage 1: Base
FROM python:3.12-slim as base

# Установка общих зависимостей
RUN pip install fastapi==0.104.1 \
    uvicorn==0.24.0 \
    psycopg2-binary==2.9.9 \
    python-jose[cryptography]==3.3.0 \
    passlib[bcrypt]==1.7.4 \
    python-multipart==0.0.6 \
    pydantic==2.5.2 \
    pydantic[email] \
    email-validator \
    python-dotenv==1.0.0 \
    sqlalchemy==2.0.23 \
    alembic==1.12.1 \
    httpx==0.25.2

WORKDIR /app
COPY utils /app/utils

# Stage 2: User Service
FROM base as user-service
COPY user_service /app/user_service
WORKDIR /app
ENV PYTHONPATH=/app
CMD ["python", "user_service/run.py"]

# Stage 3: Project Service
FROM base as project-service
COPY project_service /app/project_service
WORKDIR /app
ENV PYTHONPATH=/app
CMD ["python", "project_service/run.py"]

# Stage 4: Task Service
FROM base as task-service
COPY task_service /app/task_service
WORKDIR /app
ENV PYTHONPATH=/app
CMD ["python", "task_service/run.py"]