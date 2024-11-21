# project-management-system/services/task_service/src/api/routes.py
from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Optional
from uuid import UUID
from models import Task, TaskCreate, TaskUpdate
from database import db
from auth import get_current_user, User

router = APIRouter()

@router.post("/tasks", response_model=Task, status_code=status.HTTP_201_CREATED)
async def create_task(task: TaskCreate, current_user: User = Depends(get_current_user)):
    """
    Описание:
      Создание новой задачи.

    Args:
        task: Данные для создания задачи
        current_user: Текущий пользователь

    Returns:
        Task: Созданная задача
    """
    # В реальном приложении здесь должна быть проверка существования проекта
    new_task = Task(
        **task.dict(),
        creator_id=current_user.id
    )
    return db.create_task(new_task)

@router.get("/tasks", response_model=List[Task])
async def read_tasks(
    project_id: Optional[UUID] = None,
    assignee_id: Optional[UUID] = None,
    current_user: User = Depends(get_current_user)
):
    """
    Описание:
      Получение списка задач с возможностью фильтрации.

    Args:
        project_id: ID проекта для фильтрации
        assignee_id: ID исполнителя для фильтрации
        current_user: Текущий пользователь

    Returns:
        List[Task]: Список задач
    """
    if project_id:
        return db.get_tasks_by_project(project_id)
    if assignee_id:
        return db.get_tasks_by_assignee(assignee_id)
    return list(db.tasks.values())

@router.get("/tasks/{task_id}", response_model=Task)
async def read_task(task_id: UUID, current_user: User = Depends(get_current_user)):
    """
    Описание:
      Получение информации о конкретной задаче.

    Args:
        task_id: ID задачи
        current_user: Текущий пользователь

    Returns:
        Task: Задача

    Raises:
        HTTPException: Если задача не найдена
    """
    task = db.get_task_by_id(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return task

@router.put("/tasks/{task_id}", response_model=Task)
async def update_task(
    task_id: UUID,
    task_update: TaskUpdate,
    current_user: User = Depends(get_current_user)
):
    """
    Описание:
      Обновление задачи.

    Args:
        task_id: ID задачи
        task_update: Данные для обновления
        current_user: Текущий пользователь

    Returns:
        Task: Обновленная задача

    Raises:
        HTTPException: Если задача не найдена
    """
    try:
        updated_task = db.update_task(task_id, task_update.dict(exclude_unset=True))
        return updated_task
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.delete("/tasks/{task_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_task(task_id: UUID, current_user: User = Depends(get_current_user)):
    """
    Описание:
      Удаление задачи.

    Args:
        task_id: ID задачи
        current_user: Текущий пользователь

    Raises:
        HTTPException: Если задача не найдена
    """
    if not db.delete_task(task_id):
        raise HTTPException(status_code=404, detail="Task not found")