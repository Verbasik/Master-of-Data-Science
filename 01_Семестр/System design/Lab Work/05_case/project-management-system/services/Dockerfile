# services/Dockerfile
# Stage 1: Base
FROM python:3.12-slim as base

# Установка общих зависимостей
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
    redis==5.0.1

WORKDIR /app
COPY services/utils /app/utils

# Stage 2: User Service
FROM base as user-service
COPY services/user_service /app/user_service
WORKDIR /app/user_service
ENV PYTHONPATH=/app
CMD ["python", "run.py"]

# Stage 3: Project Service
FROM base as project-service
COPY services/project_service /app/project_service
WORKDIR /app/project_service
ENV PYTHONPATH=/app
CMD ["python", "run.py"]

# Stage 4: Task Service
FROM base as task-service
COPY services/task_service /app/task_service
WORKDIR /app/task_service
ENV PYTHONPATH=/app
CMD ["python", "run.py"]