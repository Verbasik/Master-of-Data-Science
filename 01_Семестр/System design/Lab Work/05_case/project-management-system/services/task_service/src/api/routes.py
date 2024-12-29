# project-management-system/services/task_service/src/api/routes.py

# Импорты библиотек
from typing import List, Optional, AsyncGenerator
from uuid import UUID

# Импорты FastAPI
from fastapi import APIRouter, Depends, HTTPException, Query, status

# Логирование
import logging

# Локальные импорты
from ..services.task_service import TaskService
from ..models.api_models import TaskCreate, TaskUpdate, TaskResponse
from ..models.mongo_models import TaskStatus, TaskPriority
from ..auth import get_current_user, User
from ..database import MongoDBManager

# Логирование
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Маршрутизатор FastAPI
router = APIRouter(prefix="/tasks", tags=["tasks"])


# Dependency для получения экземпляра TaskService
async def get_task_service() -> AsyncGenerator[TaskService, None]:
    """
    Description:
        Dependency для внедрения TaskService.

    Returns:
        Экземпляр TaskService.

    Raises:
        HTTPException: В случае ошибок подключения к MongoDB.

    Examples:
        Используется в маршрутах FastAPI:
        >>> @router.get("/")
        >>> async def get_tasks(service: TaskService = Depends(get_task_service)):
        >>>     return await service.get_tasks()
    """
    db_manager = None
    try:
        # Создаем экземпляр и сохраняем его
        db_manager = MongoDBManager()
        # Подключаемся и сохраняем результат
        connected_manager = await db_manager.connect()
        logger.info("Подключение к MongoDB успешно")

        service = TaskService(connected_manager)
        logger.info("TaskService успешно инициализирован")

        yield service

    except Exception as e:
        logger.error(f"Не удалось инициализировать TaskService: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ошибка инициализации сервиса задач."
        )

    finally:
        if db_manager:
            logger.info("Закрытие подключения к MongoDB")
            await db_manager.disconnect()
            logger.info("Подключение к MongoDB закрыто")

@router.post("", response_model=TaskResponse, status_code=status.HTTP_201_CREATED)
async def create_task(
    task: TaskCreate,
    current_user: User = Depends(get_current_user),
    task_service: TaskService = Depends(get_task_service)
) -> TaskResponse:
    """
    Description:
        Создание новой задачи.

    Args:
        task (TaskCreate): Данные для создания задачи
        current_user (User): Текущий пользователь
        task_service (TaskService): Сервис для работы с задачами

    Returns:
        TaskResponse: Созданная задача

    Raises:
        HTTPException: При ошибке создания задачи
    """
    try:
        logger.info(f"Creating new task by user {current_user.id}")
        logger.info(f"Task create data: {task.model_dump_json(indent=2)}")
        
        mongo_task = await task_service.create_task(task, current_user.id)
        logger.info(f"Task created successfully: {mongo_task.model_dump_json(indent=2)}")
        
        response = TaskResponse.from_mongo(mongo_task)
        logger.info(f"Prepared response: {response.model_dump_json(indent=2)}")
        
        return response
    except Exception as e:
        logger.error(f"Error creating task: {str(e)}")
        logger.exception(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating task: {str(e)}"
        )

@router.get("", response_model=List[TaskResponse])
async def get_tasks(
    project_id: Optional[UUID] = Query(None, description="Filter tasks by project ID"),
    assignee_id: Optional[UUID] = Query(None, description="Filter tasks by assignee ID"),
    status: Optional[TaskStatus] = Query(None, description="Filter tasks by status"),
    priority: Optional[TaskPriority] = Query(None, description="Filter tasks by priority"),
    current_user: User = Depends(get_current_user),
    task_service: TaskService = Depends(get_task_service)
) -> List[TaskResponse]:
    """
    Description:
        Получение списка задач с возможностью фильтрации.

    Args:
        project_id (UUID, optional): ID проекта для фильтрации
        assignee_id (UUID, optional): ID исполнителя для фильтрации
        status (TaskStatus, optional): Статус задачи для фильтрации
        priority (TaskPriority, optional): Приоритет задачи для фильтрации
        current_user (User): Текущий пользователь
        task_service (TaskService): Сервис для работы с задачами

    Returns:
        List[TaskResponse]: Список задач
    """
    try:
        criteria = {}
        if project_id:
            criteria["project_id"] = project_id
        if assignee_id:
            criteria["assignee_id"] = assignee_id
        if status:
            criteria["status"] = status
        if priority:
            criteria["priority"] = priority

        tasks = await task_service.get_tasks_by_criteria(criteria)
        return [TaskResponse.from_mongo(task) for task in tasks]
    except Exception as e:
        logger.error(f"Error fetching tasks: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching tasks: {str(e)}"
        )

@router.get("/{task_id}", response_model=TaskResponse)
async def get_task(
    task_id: UUID,
    current_user: User = Depends(get_current_user),
    task_service: TaskService = Depends(get_task_service)
) -> TaskResponse:
    """
    Description:
        Получение задачи по ID.

    Args:
        task_id (UUID): ID задачи
        current_user (User): Текущий пользователь
        task_service (TaskService): Сервис для работы с задачами

    Returns:
        TaskResponse: Задача

    Raises:
        HTTPException: Если задача не найдена
    """
    try:
        task = await task_service.get_task(task_id)
        if task is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task {task_id} not found"
            )
        return TaskResponse.from_mongo(task)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching task {task_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching task: {str(e)}"
        )

@router.put("/{task_id}", response_model=TaskResponse)
async def update_task(
    task_id: UUID,
    task_update: TaskUpdate,
    current_user: User = Depends(get_current_user),
    task_service: TaskService = Depends(get_task_service)
) -> TaskResponse:
    """
    Description:
        Обновление задачи.

    Args:
        task_id (UUID): ID задачи
        task_update (TaskUpdate): Данные для обновления
        current_user (User): Текущий пользователь
        task_service (TaskService): Сервис для работы с задачами

    Returns:
        TaskResponse: Обновленная задача

    Raises:
        HTTPException: Если задача не найдена или произошла ошибка обновления
    """
    try:
        # Проверяем существование задачи
        existing_task = await task_service.get_task(task_id)
        if existing_task is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task {task_id} not found"
            )

        # Проверяем права доступа
        if (existing_task.creator_id != current_user.id and 
            existing_task.assignee_id != current_user.id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to update this task"
            )

        logger.info(f"Updating task {task_id} by user {current_user.id}")
        updated_task = await task_service.update_task(task_id, task_update)
        if updated_task is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update task"
            )
        logger.info(f"Task {task_id} updated successfully")
        return TaskResponse.from_mongo(updated_task)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating task {task_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating task: {str(e)}"
        )

@router.delete("/{task_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_task(
    task_id: UUID,
    current_user: User = Depends(get_current_user),
    task_service: TaskService = Depends(get_task_service)
):
    """
    Description:
        Удаление задачи.

    Args:
        task_id (UUID): ID задачи
        current_user (User): Текущий пользователь
        task_service (TaskService): Сервис для работы с задачами

    Raises:
        HTTPException: Если задача не найдена или нет прав на удаление
    """
    try:
        # Проверяем существование задачи
        existing_task = await task_service.get_task(task_id)
        if existing_task is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task {task_id} not found"
            )

        # Проверяем права доступа
        if existing_task.creator_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to delete this task"
            )

        logger.info(f"Deleting task {task_id} by user {current_user.id}")
        success = await task_service.delete_task(task_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete task"
            )
        logger.info(f"Task {task_id} deleted successfully")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting task {task_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting task: {str(e)}"
        )

@router.patch("/{task_id}/status", response_model=TaskResponse)
async def update_task_status(
    task_id: UUID,
    status: TaskStatus,
    current_user: User = Depends(get_current_user),
    task_service: TaskService = Depends(get_task_service)
) -> TaskResponse:
    """
    Description:
        Обновление статуса задачи.

    Args:
        task_id (UUID): ID задачи
        status (TaskStatus): Новый статус
        current_user (User): Текущий пользователь
        task_service (TaskService): Сервис для работы с задачами

    Returns:
        TaskResponse: Обновленная задача

    Raises:
        HTTPException: При ошибке обновления статуса
    """
    try:
        # Проверяем существование задачи
        existing_task = await task_service.get_task(task_id)
        if existing_task is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task {task_id} not found"
            )

        # Проверяем права доступа
        if (existing_task.creator_id != current_user.id and 
            existing_task.assignee_id != current_user.id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to update this task status"
            )

        logger.info(f"Updating status of task {task_id} to {status} by user {current_user.id}")
        task_update = TaskUpdate(status=status)
        updated_task = await task_service.update_task(task_id, task_update)
        if updated_task is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update task status"
            )
        logger.info(f"Task {task_id} status updated successfully")
        return TaskResponse.from_mongo(updated_task)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating task status {task_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating task status: {str(e)}"
        )

@router.patch("/{task_id}/assignee", response_model=TaskResponse)
async def update_task_assignee(
    task_id: UUID,
    assignee_id: Optional[UUID],
    current_user: User = Depends(get_current_user),
    task_service: TaskService = Depends(get_task_service)
) -> TaskResponse:
    """
    Description:
        Обновление исполнителя задачи.

    Args:
        task_id (UUID): ID задачи
        assignee_id (UUID, optional): ID нового исполнителя
        current_user (User): Текущий пользователь
        task_service (TaskService): Сервис для работы с задачами

    Returns:
        TaskResponse: Обновленная задача

    Raises:
        HTTPException: При ошибке обновления исполнителя
    """
    try:
        # Проверяем существование задачи
        existing_task = await task_service.get_task(task_id)
        if existing_task is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task {task_id} not found"
            )

        # Проверяем права доступа
        if existing_task.creator_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to update task assignee"
            )

        logger.info(f"Updating assignee of task {task_id} to {assignee_id} by user {current_user.id}")
        task_update = TaskUpdate(assignee_id=assignee_id)
        updated_task = await task_service.update_task(task_id, task_update)
        if updated_task is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update task assignee"
            )
        logger.info(f"Task {task_id} assignee updated successfully")
        return TaskResponse.from_mongo(updated_task)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating task assignee {task_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating task assignee: {str(e)}"
        )