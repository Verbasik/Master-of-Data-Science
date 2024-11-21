#!/bin/bash

# Цвета для вывода
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color
BLUE='\033[0;34m'

# Функция для красивого вывода
print_step() {
   echo -e "\n${BLUE}=== $1 ===${NC}"
}

print_success() {
   echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
   echo -e "${RED}✗ $1${NC}"
}

# Параметры сервисов
USER_SERVICE="http://localhost:8000"
PROJECT_SERVICE="http://localhost:8001"
TASK_SERVICE="http://localhost:8002"

# Очистка предыдущих результатов
echo "" > test_results.log

# 1. Создание пользователя
print_step "Creating user"
USER_RESPONSE=$(curl -s -X POST \
   -H "Content-Type: application/json" \
   -d '{
       "username": "testuser",
       "email": "test@example.com",
       "password": "testpass"
   }' \
   ${USER_SERVICE}/users)

print_success "User created"
echo "User Response: $USER_RESPONSE" >> test_results.log

# 2. Получение токена
print_step "Getting authentication token"
TOKEN_RESPONSE=$(curl -s -X POST \
   -H "Content-Type: application/x-www-form-urlencoded" \
   -d "username=testuser&password=testpass" \
   ${USER_SERVICE}/token)

# Извлекаем токен с помощью простой обработки строк
TOKEN=$(echo $TOKEN_RESPONSE | grep -o '"access_token":"[^"]*' | sed 's/"access_token":"//')

if [ ! -z "$TOKEN" ]; then
   print_success "Token received successfully"
   echo "Token: $TOKEN" >> test_results.log
else
   print_error "Failed to get token"
   echo "Failed Token Response: $TOKEN_RESPONSE" >> test_results.log
   exit 1
fi

# 3. Создание проекта
print_step "Creating project"
PROJECT_RESPONSE=$(curl -s -X POST \
   -H "Content-Type: application/json" \
   -H "Authorization: Bearer $TOKEN" \
   -d '{
       "name": "Test Project",
       "description": "Test project description"
   }' \
   ${PROJECT_SERVICE}/projects)

# Извлекаем ID проекта с помощью простой обработки строк
PROJECT_ID=$(echo $PROJECT_RESPONSE | grep -o '"id":"[^"]*' | sed 's/"id":"//')

if [ ! -z "$PROJECT_ID" ]; then
   print_success "Project created with ID: $PROJECT_ID"
   echo "Project Response: $PROJECT_RESPONSE" >> test_results.log
else
   print_error "Failed to create project"
   echo "Failed Project Response: $PROJECT_RESPONSE" >> test_results.log
   exit 1
fi

# 4. Создание задачи
print_step "Creating task"
TASK_RESPONSE=$(curl -s -X POST \
   -H "Content-Type: application/json" \
   -H "Authorization: Bearer $TOKEN" \
   -d "{
       \"title\": \"Test Task\",
       \"description\": \"Test task description\",
       \"project_id\": \"$PROJECT_ID\",
       \"priority\": \"medium\"
   }" \
   ${TASK_SERVICE}/tasks)

# Извлекаем ID задачи с помощью простой обработки строк
TASK_ID=$(echo $TASK_RESPONSE | grep -o '"id":"[^"]*' | sed 's/"id":"//')

if [ ! -z "$TASK_ID" ]; then
   print_success "Task created with ID: $TASK_ID"
   echo "Task Response: $TASK_RESPONSE" >> test_results.log
else
   print_error "Failed to create task"
   echo "Failed Task Response: $TASK_RESPONSE" >> test_results.log
   exit 1
fi

# 5. Проверка созданных данных
print_step "Verifying created data"

# Проверка проекта
print_step "Getting project details"
PROJECT_GET_RESPONSE=$(curl -s -X GET \
   -H "Authorization: Bearer $TOKEN" \
   "${PROJECT_SERVICE}/projects/${PROJECT_ID}")

echo "Project Details: $PROJECT_GET_RESPONSE" >> test_results.log
print_success "Got project details"

# Проверка задачи
print_step "Getting task details"
TASK_GET_RESPONSE=$(curl -s -X GET \
   -H "Authorization: Bearer $TOKEN" \
   "${TASK_SERVICE}/tasks/${TASK_ID}")

echo "Task Details: $TASK_GET_RESPONSE" >> test_results.log
print_success "Got task details"

# 6. Проверка списков
print_step "Getting lists of projects and tasks"

# Получение списка проектов
PROJECTS_LIST=$(curl -s -X GET \
   -H "Authorization: Bearer $TOKEN" \
   ${PROJECT_SERVICE}/projects)

echo "Projects List: $PROJECTS_LIST" >> test_results.log
print_success "Got projects list"

# Получение списка задач
TASKS_LIST=$(curl -s -X GET \
   -H "Authorization: Bearer $TOKEN" \
   "${TASK_SERVICE}/tasks?project_id=${PROJECT_ID}")

echo "Tasks List: $TASKS_LIST" >> test_results.log
print_success "Got tasks list"

print_step "Test Summary"
echo -e "${GREEN}All tests completed successfully!${NC}"
echo "Detailed results are saved in test_results.log"