#!/bin/bash

# Цвета для вывода
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'

# Функции для вывода
print_step() {
    echo -e "\n${BLUE}=== $1 ===${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
    echo "Error: $1" >> test_results.log
    if [ "$2" != "no-exit" ]; then
        exit 1
    fi
}

print_warning() {
    echo -e "${YELLOW}! $1${NC}"
}

# Параметры сервисов
USER_SERVICE="http://localhost:8000"
PROJECT_SERVICE="http://localhost:8001"
TASK_SERVICE="http://localhost:8002"

# Проверка наличия .env файла
print_step "Checking environment configuration"
if [ ! -f .env ]; then
    print_error ".env file not found"
fi

# Чтение DATABASE_URL из .env
DATABASE_URL=$(grep DATABASE_URL .env | cut -d '=' -f2 | tr -d '"' | tr -d ' ')
if [ -z "$DATABASE_URL" ]; then
    print_error "DATABASE_URL not found in .env file"
fi

# Парсинг DATABASE_URL
PGUSER=$(echo $DATABASE_URL | sed -n 's/.*:\/\/\([^:]*\):.*/\1/p')
PGPASSWORD=$(echo $DATABASE_URL | sed -n 's/.*:\/\/[^:]*:\([^@]*\)@.*/\1/p')
PGHOST=$(echo $DATABASE_URL | sed -n 's/.*@\([^:]*\):.*/\1/p')
PGPORT=$(echo $DATABASE_URL | sed -n 's/.*:\([0-9]*\)\/.*/\1/p')
PGDATABASE=$(echo $DATABASE_URL | sed -n 's/.*\/\(.*\)/\1/p')

# Создание временного файла PGPASS
PGPASS_FILE=$(mktemp)
chmod 600 $PGPASS_FILE
echo "$PGHOST:$PGPORT:$PGDATABASE:$PGUSER:$PGPASSWORD" > $PGPASS_FILE
export PGPASSFILE=$PGPASS_FILE

# Инициализация лог-файла
echo "Test Results - $(date)" > test_results.log
echo "===================" >> test_results.log
echo "Testing environment:" >> test_results.log
echo "- Host: $PGHOST" >> test_results.log
echo "- Port: $PGPORT" >> test_results.log
echo "- Database: $PGDATABASE" >> test_results.log
echo "- User: $PGUSER" >> test_results.log
echo "===================" >> test_results.log

# Проверка доступности сервисов
print_step "Checking services availability"

check_service() {
    local service_url=$1
    local service_name=$2
    if curl -s -f "$service_url/docs" > /dev/null; then
        print_success "$service_name is available"
        return 0
    else
        print_error "$service_name is not available" "no-exit"
        return 1
    fi
}

check_service $USER_SERVICE "User Service"
check_service $PROJECT_SERVICE "Project Service"
check_service $TASK_SERVICE "Task Service"

# Проверка подключения к PostgreSQL
print_step "Checking PostgreSQL connection"
if ! psql -h $PGHOST -p $PGPORT -U $PGUSER -d $PGDATABASE -c '\q' 2>/dev/null; then
    print_error "Cannot connect to PostgreSQL"
fi
print_success "PostgreSQL connection successful"

# Функции проверки БД
check_table() {
    local table=$1
    local query="SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = '$table');"
    local exists=$(psql -h $PGHOST -p $PGPORT -U $PGUSER -d $PGDATABASE -tA -c "$query")
    if [ "$exists" = "f" ]; then
        print_error "Table $table does not exist" "no-exit"
        return 1
    fi
    print_success "Table $table exists"
    return 0
}

check_index() {
    local table=$1
    local column=$2
    local query="SELECT EXISTS (SELECT FROM pg_indexes WHERE tablename = '$table' AND indexdef LIKE '%$column%');"
    local exists=$(psql -h $PGHOST -p $PGPORT -U $PGUSER -d $PGDATABASE -tA -c "$query")
    if [ "$exists" = "f" ]; then
        print_warning "Index on $table($column) does not exist"
        return 1
    fi
    print_success "Index on $table($column) exists"
    return 0
}

# Проверка структуры БД
print_step "Checking database structure"
for table in "users" "projects" "tasks"; do
    check_table $table
done

print_step "Checking indexes"
check_index "users" "username"
check_index "users" "email"
check_index "projects" "name"
check_index "projects" "owner_id"
check_index "tasks" "title"
check_index "tasks" "status"
check_index "tasks" "project_id"

# Тестовые данные
TEST_USERNAME="testuser_$(date +%s)"
TEST_EMAIL="test_$(date +%s)@example.com"
TEST_PASSWORD="TestPass123!"

# 1. Тестирование User Service
print_step "Testing User Service"

# Создание пользователя
print_step "Creating test user"
USER_RESPONSE=$(curl -s -X POST \
    -H "Content-Type: application/json" \
    -d "{
        \"username\": \"$TEST_USERNAME\",
        \"email\": \"$TEST_EMAIL\",
        \"password\": \"$TEST_PASSWORD\"
    }" \
    ${USER_SERVICE}/users)

echo "User creation response: $USER_RESPONSE" >> test_results.log

# Проверка хеширования пароля
print_step "Verifying password hashing"
HASH_CHECK=$(psql -h $PGHOST -p $PGPORT -U $PGUSER -d $PGDATABASE -tA -c \
    "SELECT hashed_password FROM users WHERE username = '$TEST_USERNAME';")
if [[ $HASH_CHECK == *"$TEST_PASSWORD"* ]]; then
    print_error "Password is not properly hashed"
fi
print_success "Password is properly hashed"

# Получение токена
print_step "Getting authentication token"
TOKEN_RESPONSE=$(curl -s -X POST \
    -H "Content-Type: application/x-www-form-urlencoded" \
    -d "username=$TEST_USERNAME&password=$TEST_PASSWORD" \
    ${USER_SERVICE}/token)

TOKEN=$(echo $TOKEN_RESPONSE | grep -o '"access_token":"[^"]*' | sed 's/"access_token":"//')
if [ -z "$TOKEN" ]; then
    print_error "Failed to get authentication token"
fi
print_success "Authentication successful"
echo "Token received: ${TOKEN:0:20}..." >> test_results.log

# 2. Тестирование Project Service
print_step "Testing Project Service"

# Создание проекта
PROJECT_RESPONSE=$(curl -s -X POST \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $TOKEN" \
    -d "{
        \"name\": \"Test Project $(date +%s)\",
        \"description\": \"Test project description\"
    }" \
    ${PROJECT_SERVICE}/projects)

PROJECT_ID=$(echo $PROJECT_RESPONSE | grep -o '"id":"[^"]*' | sed 's/"id":"//')
if [ -z "$PROJECT_ID" ]; then
    print_error "Failed to create project"
fi
print_success "Project created successfully"
echo "Project creation response: $PROJECT_RESPONSE" >> test_results.log

# Проверка проекта в БД
PROJECT_DB_CHECK=$(psql -h $PGHOST -p $PGPORT -U $PGUSER -d $PGDATABASE -tA -c \
    "SELECT COUNT(*) FROM projects WHERE id = '$PROJECT_ID';")
if [ "$PROJECT_DB_CHECK" -ne "1" ]; then
    print_error "Project not found in database"
fi

# 3. Тестирование Task Service
print_step "Testing Task Service"

# Создание задачи
TASK_RESPONSE=$(curl -s -X POST \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $TOKEN" \
    -d "{
        \"title\": \"Test Task $(date +%s)\",
        \"description\": \"Test task description\",
        \"project_id\": \"$PROJECT_ID\",
        \"priority\": \"medium\"
    }" \
    ${TASK_SERVICE}/tasks)

TASK_ID=$(echo $TASK_RESPONSE | grep -o '"id":"[^"]*' | sed 's/"id":"//')
if [ -z "$TASK_ID" ]; then
    print_error "Failed to create task"
fi
print_success "Task created successfully"
echo "Task creation response: $TASK_RESPONSE" >> test_results.log

# Проверка задачи в БД
TASK_DB_CHECK=$(psql -h $PGHOST -p $PGPORT -U $PGUSER -d $PGDATABASE -tA -c \
    "SELECT COUNT(*) FROM tasks WHERE id = '$TASK_ID' AND project_id = '$PROJECT_ID';")
if [ "$TASK_DB_CHECK" -ne "1" ]; then
    print_error "Task not found in database or project link incorrect"
fi

# 4. Тестирование обновлений
print_step "Testing update operations"

# Обновление проекта
UPDATE_PROJECT_RESPONSE=$(curl -s -X PUT \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $TOKEN" \
    -d "{
        \"name\": \"Updated Project $(date +%s)\",
        \"description\": \"Updated project description\"
    }" \
    "${PROJECT_SERVICE}/projects/${PROJECT_ID}")

echo "Project update response: $UPDATE_PROJECT_RESPONSE" >> test_results.log

# Обновление задачи
UPDATE_TASK_RESPONSE=$(curl -s -X PUT \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $TOKEN" \
    -d "{
        \"title\": \"Updated Task $(date +%s)\",
        \"description\": \"Updated task description\",
        \"status\": \"in_progress\"
    }" \
    "${TASK_SERVICE}/tasks/${TASK_ID}")

echo "Task update response: $UPDATE_TASK_RESPONSE" >> test_results.log

# 5. Тестирование каскадного удаления
print_step "Testing cascade delete"

# Удаление проекта
curl -s -X DELETE \
    -H "Authorization: Bearer $TOKEN" \
    "${PROJECT_SERVICE}/projects/${PROJECT_ID}"

# Проверка каскадного удаления задач
TASK_CASCADE_CHECK=$(psql -h $PGHOST -p $PGPORT -U $PGUSER -d $PGDATABASE -tA -c \
    "SELECT COUNT(*) FROM tasks WHERE project_id = '$PROJECT_ID';")
if [ "$TASK_CASCADE_CHECK" -ne "0" ]; then
    print_error "Cascade delete failed - orphaned tasks exist"
fi
print_success "Cascade delete verified"

# Очистка
print_step "Cleaning up test data"
psql -h $PGHOST -p $PGPORT -U $PGUSER -d $PGDATABASE -c \
    "DELETE FROM users WHERE username = '$TEST_USERNAME';" > /dev/null 2>&1

# Удаление временного файла PGPASS
rm -f $PGPASS_FILE

print_step "Test Summary"
echo -e "${GREEN}All tests completed successfully!${NC}"
echo "Detailed results are saved in test_results.log"