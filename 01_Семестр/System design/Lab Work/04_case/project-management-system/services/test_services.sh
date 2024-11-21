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
}

print_warning() {
    echo -e "${YELLOW}! $1${NC}"
    echo "Warning: $1" >> test_results.log
}

# Функции обработки ошибок
handle_error() {
    local exit_code=$1
    local error_msg=$2
    print_error "$error_msg"
    cleanup_test_data
    exit $exit_code
}

# Функции работы с БД
parse_database_url() {
    local url=$1
    export PGUSER=$(echo $url | sed -n 's/.*:\/\/\([^:]*\):.*/\1/p')
    export PGPASSWORD=$(echo $url | sed -n 's/.*:\/\/[^:]*:\([^@]*\)@.*/\1/p')
    export PGHOST=$(echo $url | sed -n 's/.*@\([^:]*\):.*/\1/p')
    export PGPORT=$(echo $url | sed -n 's/.*:\([0-9]*\)\/.*/\1/p')
    export PGDATABASE=$(echo $url | sed -n 's/.*\/\([^?]*\).*/\1/p')
}

setup_database_connection() {
    parse_database_url "$DATABASE_URL"
    PGPASS_FILE=$(mktemp)
    chmod 600 $PGPASS_FILE
    echo "$PGHOST:$PGPORT:$PGDATABASE:$PGUSER:$PGPASSWORD" > $PGPASS_FILE
    export PGPASSFILE=$PGPASS_FILE
}

check_required_vars() {
    local missing_vars=()
    
    [ -z "$DATABASE_URL" ] && missing_vars+=("DATABASE_URL")
    [ -z "$MONGODB_URL" ] && missing_vars+=("MONGODB_URL")
    [ -z "$MONGODB_DB_NAME" ] && missing_vars+=("MONGODB_DB_NAME")
    
    if [ ${#missing_vars[@]} -ne 0 ]; then
        handle_error 1 "Missing required environment variables: ${missing_vars[*]}"
    fi
}

cleanup_test_data() {
    print_step "Cleaning up test data"
    
    # Очистка PostgreSQL
    if [ ! -z "$TEST_USERNAME" ]; then
        PGPASSFILE=$PGPASS_FILE psql -h $PGHOST -p $PGPORT -U $PGUSER -d $PGDATABASE \
            -c "DELETE FROM users WHERE username = '$TEST_USERNAME';" \
            > /dev/null 2>&1 || print_warning "Failed to cleanup PostgreSQL test user"
    fi
    
    # Очистка MongoDB
    if [ ! -z "$PROJECT_ID" ]; then
        docker exec mongodb mongosh "$MONGODB_URL" --quiet \
            --eval "db.tasks.deleteMany({project_id: '$PROJECT_ID'})" \
            > /dev/null 2>&1 || print_warning "Failed to cleanup MongoDB test tasks"
    fi
    
    # Удаление временных файлов
    [ -f "$PGPASS_FILE" ] && rm -f "$PGPASS_FILE"
}

# Проверка MongoDB
check_mongodb_connection() {
    if ! docker exec mongodb mongosh "$MONGODB_URL" --quiet --eval "db.runCommand({ping: 1}).ok" | grep -q "1"; then
        return 1
    fi
    return 0
}

check_mongo_indexes() {
    local index_check_failed=0
    
    check_index() {
        local index=$1
        local result=$(docker exec mongodb mongosh "$MONGODB_URL" --quiet \
            --eval "db.tasks.getIndexes().some(i => Object.keys(i.key).includes('$index'))")
        
        if [ "$result" != "true" ]; then
            print_warning "Index '$index' not found"
            index_check_failed=1
            return 1
        fi
        print_success "Index '$index' exists"
        return 0
    }
    
    local required_indexes=(
        "project_id"
        "creator_id"
        "assignee_id"
        "status"
        "priority"
        "created_at"
    )
    
    for index in "${required_indexes[@]}"; do
        check_index "$index"
    done
    
    # Проверка составного индекса
    local compound_result=$(docker exec mongodb mongosh "$MONGODB_URL" --quiet \
        --eval 'db.tasks.getIndexes().some(i => 
            i.key.project_id === 1 && 
            i.key.status === 1 && 
            i.key.priority === -1
        )')
    
    if [ "$compound_result" != "true" ]; then
        print_warning "Compound index not found"
        index_check_failed=1
        return 1
    fi
    print_success "Compound index exists"
    
    return $index_check_failed
}

# Инициализация
print_step "Initializing test environment"

# Проверка .env файла
if [ ! -f .env ]; then
    handle_error 1 ".env file not found"
fi

# Загрузка переменных окружения
set -a
source .env
set +a

# Проверка необходимых переменных
check_required_vars

# Настройка подключения к БД
setup_database_connection

# Проверка сервисов
print_step "Checking services availability"

for service in "user:8000" "project:8001" "task:8002"; do
    name=${service%:*}
    port=${service#*:}
    if ! curl -s -f "http://localhost:$port/health" > /dev/null; then
        handle_error 1 "$name Service is not available"
    fi
    print_success "$name Service is available"
done

# Проверка PostgreSQL
print_step "Checking PostgreSQL connection"
if ! PGPASSFILE=$PGPASS_FILE psql -h $PGHOST -p $PGPORT -U $PGUSER -d $PGDATABASE -c '\conninfo' > /dev/null 2>&1; then
    handle_error 1 "Cannot connect to PostgreSQL"
fi
print_success "PostgreSQL connection successful"

# Проверка MongoDB
print_step "Checking MongoDB connection"
if ! check_mongodb_connection; then
    handle_error 1 "Cannot connect to MongoDB"
fi
print_success "MongoDB connection successful"

# Проверка индексов MongoDB
print_step "Checking MongoDB indexes"
check_mongo_indexes

# Тестовые данные
TEST_USERNAME="testuser_$(date +%s)"
TEST_EMAIL="test_$(date +%s)@example.com"
TEST_PASSWORD="TestPass123!"

# Тестирование User Service
print_step "Testing User Service"

# Создание пользователя
USER_RESPONSE=$(curl -s -X POST \
    -H "Content-Type: application/json" \
    -d "{
        \"username\": \"$TEST_USERNAME\",
        \"email\": \"$TEST_EMAIL\",
        \"password\": \"$TEST_PASSWORD\"
    }" \
    ${USER_SERVICE_URL}/users)

echo "User creation response: $USER_RESPONSE" >> test_results.log

# Получение токена
TOKEN_RESPONSE=$(curl -s -X POST \
    -H "Content-Type: application/x-www-form-urlencoded" \
    -d "username=$TEST_USERNAME&password=$TEST_PASSWORD" \
    ${USER_SERVICE_URL}/token)

TOKEN=$(echo $TOKEN_RESPONSE | grep -o '"access_token":"[^"]*' | sed 's/"access_token":"//')
if [ -z "$TOKEN" ]; then
    handle_error 1 "Failed to get authentication token"
fi
print_success "Authentication successful"

# Тестирование Project Service
print_step "Testing Project Service"

PROJECT_RESPONSE=$(curl -s -X POST \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $TOKEN" \
    -d "{
        \"name\": \"Test Project $(date +%s)\",
        \"description\": \"Test project description\"
    }" \
    ${PROJECT_SERVICE_URL}/projects)

PROJECT_ID=$(echo $PROJECT_RESPONSE | grep -o '"id":"[^"]*' | sed 's/"id":"//')
if [ -z "$PROJECT_ID" ]; then
    handle_error 1 "Failed to create project"
fi
print_success "Project created successfully"

# Тестирование Task Service
print_step "Testing Task Service"

REQUEST_DATA="{
    \"title\": \"Test Task $(date +%s)\",
    \"description\": \"Test task description\",
    \"project_id\": \"$PROJECT_ID\",
    \"priority\": \"medium\"
}"

echo "Task creation request: $REQUEST_DATA" >> test_results.log

TASK_RESPONSE=$(curl -s -X POST \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $TOKEN" \
    -d "$REQUEST_DATA" \
    ${TASK_SERVICE_URL}/tasks)

echo "Task creation response: $TASK_RESPONSE" >> test_results.log

# Проверяем, что ответ является валидным JSON
if ! echo "$TASK_RESPONSE" | jq -e . >/dev/null 2>&1; then
    handle_error 1 "Invalid JSON response from task creation"
fi

# Проверяем наличие ошибок в ответе
if echo "$TASK_RESPONSE" | jq -e 'has("detail")' >/dev/null; then
    handle_error 1 "Task creation failed: $(echo $TASK_RESPONSE | jq -r '.detail')"
fi

# Получаем ID задачи
TASK_ID=$(echo "$TASK_RESPONSE" | jq -r '.id')
if [ -z "$TASK_ID" ] || [ "$TASK_ID" = "null" ]; then
    handle_error 1 "Failed to get task ID from response"
fi
print_success "Task created successfully with ID: $TASK_ID"

# Проверяем создание в MongoDB с повторными попытками
print_step "Verifying task in MongoDB"
check_mongo_document() {
    local collection=$1
    local document_id=$2
    local retries=5
    local wait_time=1
    
    echo "Checking MongoDB for document ID: $document_id" >> test_results.log
    
    while [ $retries -gt 0 ]; do
        # Проверим все документы в коллекции
        echo "Current documents in collection:" >> test_results.log
        local all_docs=$(docker exec mongodb mongosh "$MONGODB_URL" --quiet \
            --eval "db.$collection.find().toArray()")
        echo "$all_docs" >> test_results.log
        
        # Пробуем найти документ разными способами
        local by_id=$(docker exec mongodb mongosh "$MONGODB_URL" --quiet \
            --eval "db.$collection.findOne({_id: '$document_id'})")
        
        local by_stripped_id=$(docker exec mongodb mongosh "$MONGODB_URL" --quiet \
            --eval "db.$collection.findOne({_id: '${document_id//-/}'})")
            
        echo "Search by original ID: $by_id" >> test_results.log
        echo "Search by stripped ID: $by_stripped_id" >> test_results.log
        
        if [ "$by_id" != "null" ] || [ "$by_stripped_id" != "null" ]; then
            print_success "Document found in MongoDB"
            return 0
        fi
        
        retries=$((retries - 1))
        if [ $retries -gt 0 ]; then
            echo "Waiting for document to appear... ($retries retries left)" >> test_results.log
            sleep $wait_time
        fi
    done
    
    # Если документ не найден, выведем дополнительную информацию
    echo "MongoDB URL: $MONGODB_URL" >> test_results.log
    echo "Collection: $collection" >> test_results.log
    echo "Document ID: $document_id" >> test_results.log
    echo "Attempted ID without dashes: ${document_id//-/}" >> test_results.log
    
    handle_error 1 "Document not found in MongoDB after multiple retries"
}

# В основной части скрипта:
TASK_ID=$(echo "$TASK_RESPONSE" | jq -r '.id')
if [ -z "$TASK_ID" ] || [ "$TASK_ID" = "null" ]; then
    handle_error 1 "Failed to get task ID from response"
fi
print_success "Task created successfully with ID: $TASK_ID"

echo "Waiting for MongoDB replication..."
sleep 2  # Даем время на репликацию

check_mongo_document "tasks" "$TASK_ID"

# Регистрируем очистку при выходе
trap cleanup_test_data EXIT

print_step "Test Summary"
echo -e "${GREEN}All tests completed successfully!${NC}"
echo "Detailed results are saved in test_results.log"