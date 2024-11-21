# project-management-system/services/task_service/src/api/routes.py
from uuid import UUID
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status

from ..models.api_models import Task, TaskCreate, TaskUpdate, TaskResponse
from ..models.database_models import Task as TaskDB
from ..database import db
from ..auth import get_current_user, User

router = APIRouter()

@router.post("/tasks", response_model=TaskResponse, status_code=status.HTTP_201_CREATED)
async def create_task(task: TaskCreate, current_user: User = Depends(get_current_user)):
    """
    Description:
        Создание новой задачи.

    Args:
        task (TaskCreate): Данные для создания задачи.
        current_user (User): Текущий пользователь.

    Returns:
        TaskResponse: Созданная задача.

    Raises:
        HTTPException: Если возникла ошибка при создании задачи.

    Examples:
        >>> task = TaskCreate(title="New Task", description="Task description")
        >>> response = await create_task(task, current_user)
        >>> response.title
        'New Task'
    """
    try:
        task_data = {
            **task.dict(),
            "creator_id": current_user.id
        }
        return db.create(TaskDB, task_data)
    except Exception as e:
        print(f"Error creating task: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/tasks", response_model=List[Task])
async def read_tasks(
    project_id: Optional[UUID] = None,
    assignee_id: Optional[UUID] = None,
    current_user: User = Depends(get_current_user)
):
    """
    Description:
        Получение списка задач с возможностью фильтрации по проекту и исполнителю.

    Args:
        project_id (Optional[UUID]): ID проекта для фильтрации.
        assignee_id (Optional[UUID]): ID исполнителя для фильтрации.
        current_user (User): Текущий пользователь.

    Returns:
        List[Task]: Список задач.

    Examples:
        >>> tasks = await read_tasks(project_id=UUID('...'), current_user=current_user)
        >>> len(tasks)
        3
    """
    if project_id:
        return db.get_tasks_by_project(project_id)
    if assignee_id:
        return db.get_tasks_by_assignee(assignee_id)
    return list(db.tasks.values())

@router.get("/tasks/{task_id}", response_model=Task)
async def read_task(task_id: UUID, current_user: User = Depends(get_current_user)):
    """
    Description:
        Получение информации о конкретной задаче.

    Args:
        task_id (UUID): ID задачи.
        current_user (User): Текущий пользователь.

    Returns:
        Task: Задача.

    Raises:
        HTTPException: Если задача не найдена.

    Examples:
        >>> task = await read_task(UUID('...'), current_user)
        >>> task.title
        'Existing Task'
    """
    task = db.get(TaskDB, task_id)
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
    Description:
        Обновление задачи.

    Args:
        task_id (UUID): ID задачи.
        task_update (TaskUpdate): Данные для обновления.
        current_user (User): Текущий пользователь.

    Returns:
        Task: Обновленная задача.

    Raises:
        HTTPException: Если задача не найдена.

    Examples:
        >>> updated_task = await update_task(UUID('...'), TaskUpdate(title="Updated Title"), current_user)
        >>> updated_task.title
        'Updated Title'
    """
    try:
        updated_task = db.update(TaskDB, task_id, task_update.dict(exclude_unset=True))
        return updated_task
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.delete("/tasks/{task_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_task(task_id: UUID, current_user: User = Depends(get_current_user)):
    """
    Description:
        Удаление задачи.

    Args:
        task_id (UUID): ID задачи.
        current_user (User): Текущий пользователь.

    Raises:
        HTTPException: Если задача не найдена.

    Examples:
        >>> await delete_task(UUID('...'), current_user)
    """
    if not db.delete(TaskDB, task_id):
        raise HTTPException(status_code=404, detail="Task not found")
