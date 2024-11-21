# project-management-system/services/project_service/src/api/routes.py
from fastapi import APIRouter, Depends, HTTPException, status
from typing import List
from uuid import UUID
from models import Project, ProjectCreate, ProjectInDB
from database import db
from auth import get_current_user, User

router = APIRouter()

@router.post("/projects", response_model=Project, status_code=status.HTTP_201_CREATED)
async def create_project(project: ProjectCreate, current_user: User = Depends(get_current_user)):
    """
    Описание:
      Создание нового проекта.

    Аргументы:
        project (ProjectCreate): Данные для создания проекта.
        current_user (User): Текущий аутентифицированный пользователь.

    Возвращает:
        Project: Созданный объект проекта.

    Исключения:
        HTTPException: Если пользователь не аутентифицирован.

    Пример использования:
        >>> response = client.post("/projects", json={"name": "Новый проект", "description": "Описание проекта"})
        >>> response.status_code
        201
        >>> response.json()["name"]
        'Новый проект'
    """
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    project_in_db = ProjectInDB(**project.dict(), owner_id=current_user.id)
    created_project = db.create_project(project_in_db)
    return created_project

@router.get("/projects", response_model=List[Project])
async def read_projects(current_user: User = Depends(get_current_user)):
    """
    Описание:
      Получение списка проектов текущего пользователя.

    Аргументы:
        current_user (User): Текущий аутентифицированный пользователь.

    Возвращает:
        List[Project]: Список проектов пользователя.

    Исключения:
        HTTPException: Если пользователь не аутентифицирован.

    Пример использования:
        >>> response = client.get("/projects")
        >>> response.status_code
        200
        >>> len(response.json()) > 0
        True
    """
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    return db.get_projects_by_owner(current_user.id)

@router.get("/projects/{project_id}", response_model=Project)
async def read_project(project_id: UUID, current_user: User = Depends(get_current_user)):
    """
    Описание:
      Получение информации о конкретном проекте.

    Аргументы:
        project_id (UUID): ID проекта.
        current_user (User): Текущий аутентифицированный пользователь.

    Возвращает:
        Project: Объект проекта.

    Исключения:
        HTTPException: Если проект не найден или пользователь не имеет прав доступа.

    Пример использования:
        >>> project = db.create_project(ProjectInDB(name="Тестовый проект", description="Описание", owner_id=current_user.id))
        >>> response = client.get(f"/projects/{project.id}")
        >>> response.status_code
        200
        >>> response.json()["name"]
        'Тестовый проект'
    """
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    project = db.get_project_by_id(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if project.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    
    return project

@router.put("/projects/{project_id}", response_model=Project)
async def update_project(project_id: UUID, project: ProjectCreate, current_user: User = Depends(get_current_user)):
    """
    Описание:
      Обновление информации о проекте.

    Аргументы:
        project_id (UUID): ID проекта.
        project (ProjectCreate): Новые данные проекта.
        current_user (User): Текущий аутентифицированный пользователь.

    Возвращает:
        Project: Обновленный объект проекта.

    Исключения:
        HTTPException: Если проект не найден или пользователь не имеет прав доступа.

    Пример использования:
        >>> project = db.create_project(ProjectInDB)
        (name="Тестовый проект", description="Описание", owner_id=current_user.id))
        >>> response = client.put(f"/projects/{project.id}", json={"name": "Обновленный проект", "description": "Новое описание"})
        >>> response.status_code
        200
        >>> response.json()["name"]
        'Обновленный проект'
    """
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    existing_project = db.get_project_by_id(project_id)
    if not existing_project:
        raise HTTPException(status_code=404, detail="Project not found")
    if existing_project.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    
    updated_project = ProjectInDB(**project.dict(), id=project_id, owner_id=current_user.id)
    return db.update_project(updated_project)

@router.delete("/projects/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project(project_id: UUID, current_user: User = Depends(get_current_user)):
    """
    Описание:
      Удаление проекта.

    Аргументы:
        project_id (UUID): ID проекта для удаления.
        current_user (User): Текущий аутентифицированный пользователь.

    Возвращает:
        None

    Исключения:
        HTTPException: Если проект не найден или пользователь не имеет прав доступа.

    Пример использования:
        >>> project = db.create_project(ProjectInDB(name="Проект для удаления", description="Описание", owner_id=current_user.id))
        >>> response = client.delete(f"/projects/{project.id}")
        >>> response.status_code
        204
    """
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    existing_project = db.get_project_by_id(project_id)
    if not existing_project:
        raise HTTPException(status_code=404, detail="Project not found")
    if existing_project.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    
    db.delete_project(project_id)

@router.get("/projects/search", response_model=List[Project])
async def search_projects(name: str, current_user: User = Depends(get_current_user)):
    """
    Описание:
      Поиск проектов по имени.

    Аргументы:
        name (str): Строка для поиска в названии проекта.
        current_user (User): Текущий аутентифицированный пользователь.

    Возвращает:
        List[Project]: Список проектов, соответствующих критерию поиска.

    Исключения:
        HTTPException: Если пользователь не аутентифицирован.

    Пример использования:
        >>> db.create_project(ProjectInDB(name="Проект А", description="Описание", owner_id=current_user.id))
        >>> db.create_project(ProjectInDB(name="Проект Б", description="Описание", owner_id=current_user.id))
        >>> response = client.get("/projects/search?name=Проект")
        >>> response.status_code
        200
        >>> len(response.json()) == 2
        True
    """
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    user_projects = db.get_projects_by_owner(current_user.id)
    return [project for project in user_projects if name.lower() in project.name.lower()]