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

# Функции для проверки Kafka
check_kafka_connection() {
    echo "Checking Kafka connection at ${KAFKA_EXTERNAL_BOOTSTRAP_SERVERS}..."
    if ! nc -zv localhost 9092 2>/dev/null; then
        echo "Failed to connect to Kafka at ${KAFKA_EXTERNAL_BOOTSTRAP_SERVERS}"
        return 1
    fi
    return 0
}

ensure_kafka_topic() {
    print_step "Ensuring Kafka topics"
    echo "Creating topic: ${KAFKA_TASK_TOPIC}"
    
    echo "Attempting to create topic using localhost:9092..."
    docker exec kafka /usr/bin/kafka-topics \
        --bootstrap-server localhost:9092 \
        --create \
        --if-not-exists \
        --topic "${KAFKA_TASK_TOPIC}" \
        --partitions 1 \
        --replication-factor 1 \
        2>&1 || {
        local err=$?
        print_warning "Failed to create topic using localhost:9092"
        
        echo "Attempting to create topic using kafka:9092..."
        docker exec kafka /usr/bin/kafka-topics \
            --bootstrap-server kafka:9092 \
            --create \
            --if-not-exists \
            --topic "${KAFKA_TASK_TOPIC}" \
            --partitions 1 \
            --replication-factor 1 \
            2>&1 || {
            print_error "Failed to create topic using both addresses"
            return 1
        }
    }
    
    # Проверяем, что топик действительно создан
    docker exec kafka /usr/bin/kafka-topics \
        --bootstrap-server localhost:9092 \
        --list | grep -q "${KAFKA_TASK_TOPIC}" || {
        print_error "Topic was not created successfully"
        return 1
    }
    
    print_success "Topic ${KAFKA_TASK_TOPIC} created/verified"
    return 0
}

check_kafka_topics() {
    if ! docker exec kafka /usr/bin/kafka-topics \
        --bootstrap-server ${KAFKA_INTERNAL_BOOTSTRAP_SERVERS} \
        --list | grep -q "${KAFKA_TASK_TOPIC}"; then
        echo "Topic ${KAFKA_TASK_TOPIC} not found"
        return 1
    fi
    return 0
}

monitor_kafka_events() {
    local task_id=$1
    local max_wait=30
    local counter=0
    
    print_step "Monitoring Kafka events for task: $task_id"
    
    while [ $counter -lt $max_wait ]; do
        echo "Checking Kafka events (attempt $((counter + 1))/$max_wait)..."
        if docker exec kafka /usr/bin/kafka-console-consumer \
            --bootstrap-server kafka:9092 \
            --topic "$KAFKA_TASK_TOPIC" \
            --from-beginning \
            --max-messages 1 | grep -q "$task_id"; then
            
            print_success "Task event found in Kafka"
            return 0
        fi
        
        counter=$((counter + 1))
        sleep 1
    done
    
    print_error "Task event not found in Kafka after $max_wait seconds"
    return 1
}

# Проверка переменных окружения
check_required_vars() {
    echo "Checking environment variables..."
    local missing_vars=()
    
    # Основные переменные
    [ -z "$DATABASE_URL" ] && missing_vars+=("DATABASE_URL")
    [ -z "$MONGODB_URL" ] && missing_vars+=("MONGODB_URL")
    [ -z "$MONGODB_DB_NAME" ] && missing_vars+=("MONGODB_DB_NAME")
    
    # Настройка Kafka адресов
    export KAFKA_EXTERNAL_BOOTSTRAP_SERVERS="localhost:9092"
    export KAFKA_INTERNAL_BOOTSTRAP_SERVERS="kafka:29092"
    export KAFKA_BOOTSTRAP_SERVERS="kafka:29092"
    export KAFKA_TASK_TOPIC="task-events"
    
    # Redis переменные
    [ -z "$REDIS_PASSWORD" ] && export REDIS_PASSWORD="redispass"
    [ -z "$REDIS_HOST" ] && export REDIS_HOST="redis"
    [ -z "$REDIS_PORT" ] && export REDIS_PORT="6379"
    
    if [ ${#missing_vars[@]} -ne 0 ]; then
        handle_error 1 "Missing required environment variables: ${missing_vars[*]}"
    fi
}

# Функции очистки
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
    
    # Очистка Redis
    if [ ! -z "$TEST_USERNAME" ]; then
        docker exec redis redis-cli -a "$REDIS_PASSWORD" DEL "user:$TEST_USERNAME" \
            > /dev/null 2>&1 || print_warning "Failed to cleanup Redis test data"
    fi
    
    # Очистка Kafka
    if [ ! -z "$KAFKA_TASK_TOPIC" ]; then
        docker exec kafka /usr/bin/kafka-topics \
            --bootstrap-server kafka:9092 \
            --delete --topic "$KAFKA_TASK_TOPIC" \
            > /dev/null 2>&1 || print_warning "Failed to cleanup Kafka topic"
    fi
    
    # Удаление временных файлов
    [ -f "$PGPASS_FILE" ] && rm -f "$PGPASS_FILE"
}

# Проверка сервисов
check_services_health() {
    print_step "Checking services availability"
    
    # Проверка основных сервисов
    for service in "user:8000" "project:8001" "task:8002"; do
        name=${service%:*}
        port=${service#*:}
        if ! curl -s -f "http://localhost:$port/health" > /dev/null; then
            handle_error 1 "$name Service is not available"
        fi
        print_success "$name Service is available"
    done
    
    # Проверка Kafka
    print_step "Checking Kafka health"
    if ! check_kafka_connection; then
        handle_error 1 "Cannot connect to Kafka"
    fi
    print_success "Kafka connection successful"
    
    # Создание/проверка топиков
    if ! ensure_kafka_topic; then
        handle_error 1 "Failed to ensure Kafka topics"
    fi
    
    if ! check_kafka_topics; then
        handle_error 1 "Required Kafka topics not found"
    fi
    print_success "Kafka topics verified"
}

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
check_services_health

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

USER_ID=$(echo $USER_RESPONSE | grep -o '"id":"[^"]*' | sed 's/"id":"//')
if [ -z "$USER_ID" ]; then
    handle_error 1 "Failed to create user"
fi

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

# Тестирование Task Service (CQRS)
test_task_creation

# Регистрируем очистку при выходе
trap cleanup_test_data EXIT

print_step "Test Summary"
echo -e "${GREEN}All tests completed successfully!${NC}"
echo "Detailed results are saved in test_results.log"