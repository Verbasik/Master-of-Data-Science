# project-management-system/services/project_service/src/api/routes.py
from uuid import UUID
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status

# Импорт моделей
from ..models.database_models import Project as ProjectDB
from ..models.api_models import (
    ProjectCreate,
    Project as ProjectResponse
)
from ..database import db
from ..auth import get_current_user, User

router = APIRouter()

@router.post("/projects", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
async def create_project(project: ProjectCreate, current_user: User = Depends(get_current_user)) -> ProjectResponse:
    """
    Description:
        Создание нового проекта.

    Args:
        project (ProjectCreate): Данные проекта для создания.
        current_user (User): Аутентифицированный пользователь.

    Returns:
        ProjectResponse: Данные созданного проекта.

    Raises:
        HTTPException: При ошибках создания проекта.
    """
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated"
        )

    project_data = {**project.dict(), "owner_id": current_user.id}

    try:
        return db.create(ProjectDB, project_data)
    except Exception as e:
        print(f"Error creating project: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error creating project"
        )

@router.get("/projects", response_model=List[ProjectResponse])
async def read_projects(current_user: User = Depends(get_current_user)) -> List[ProjectResponse]:
    """
    Description:
        Получение списка проектов текущего пользователя.

    Args:
        current_user (User): Аутентифицированный пользователь.

    Returns:
        List[ProjectResponse]: Список проектов пользователя.

    Raises:
        HTTPException: При ошибках получения проектов.
    """
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated"
        )

    try:
        projects = db.get_multi(ProjectDB)
        return [p for p in projects if p.owner_id == current_user.id]
    except Exception as e:
        print(f"Error getting projects: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving projects"
        )

@router.get("/projects/{project_id}", response_model=ProjectResponse)
async def read_project(project_id: UUID, current_user: User = Depends(get_current_user)) -> ProjectResponse:
    """
    Description:
        Получение информации о конкретном проекте.

    Args:
        project_id (UUID): Идентификатор проекта.
        current_user (User): Аутентифицированный пользователь.

    Returns:
        ProjectResponse: Данные проекта.

    Raises:
        HTTPException: При ошибках получения проекта.
    """
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated"
        )

    try:
        project = db.get(ProjectDB, project_id)
        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Project not found"
            )
        if project.owner_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not enough permissions"
            )
        
        return project
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting project: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving project"
        )

@router.put("/projects/{project_id}", response_model=ProjectResponse)
async def update_project(
    project_id: UUID,
    project: ProjectCreate,
    current_user: User = Depends(get_current_user)
) -> ProjectResponse:
    """
    Description:
        Обновление информации о проекте.

    Args:
        project_id (UUID): Идентификатор проекта.
        project (ProjectCreate): Данные для обновления проекта.
        current_user (User): Аутентифицированный пользователь.

    Returns:
        ProjectResponse: Обновленные данные проекта.

    Raises:
        HTTPException: При ошибках обновления проекта.
    """
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated"
        )

    try:
        existing_project = db.get(ProjectDB, project_id)
        if not existing_project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Project not found"
            )
        if existing_project.owner_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not enough permissions"
            )

        update_data = project.dict(exclude_unset=True)
        return db.update(ProjectDB, project_id, update_data)
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error updating project: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error updating project"
        )

@router.delete("/projects/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project(project_id: UUID, current_user: User = Depends(get_current_user)) -> None:
    """
    Description:
        Удаление проекта.

    Args:
        project_id (UUID): Идентификатор проекта.
        current_user (User): Аутентифицированный пользователь.

    Returns:
        None

    Raises:
        HTTPException: При ошибках удаления проекта.
    """
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated"
        )

    try:
        existing_project = db.get(ProjectDB, project_id)
        if not existing_project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Project not found"
            )
        if existing_project.owner_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not enough permissions"
            )

        db.delete(ProjectDB, project_id)
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error deleting project: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error deleting project"
        )

@router.get("/projects/search", response_model=List[ProjectResponse])
async def search_projects(name: str, current_user: User = Depends(get_current_user)) -> List[ProjectResponse]:
    """
    Description:
        Поиск проектов по имени.

    Args:
        name (str): Имя для поиска проектов.
        current_user (User): Аутентифицированный пользователь.

    Returns:
        List[ProjectResponse]: Список найденных проектов.

    Raises:
        HTTPException: При ошибках поиска проектов.
    """
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated"
        )

    try:
        projects = db.get_multi(ProjectDB)
        user_projects = [p for p in projects if p.owner_id == current_user.id]
        return [p for p in user_projects if name.lower() in p.name.lower()]
    except Exception as e:
        print(f"Error searching projects: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error searching projects"
        )