// Определяем рабочее пространство системы управления проектами
workspace {
    // Модель системы, включающая в себя элементы и их взаимодействия
    model {
        // Определяем роль пользователя системы
        user = person "Пользователь" "Пользователь системы управления проектами"

        // Определяем программную систему "Система управления проектами"
        projectManagementSystem = softwareSystem "Система управления проектами" "Позволяет управлять проектами, задачами и пользователями" {
            // Контейнеры по принципу bounded contexts в DDD

            // Сервис управления проектами
            projectService = container "Project Service" "Управляет проектами" "Java, Spring Boot"
            // Сервис управления задачами
            taskService = container "Task Service" "Управляет задачами" "Java, Spring Boot"
            // Сервис управления пользователями
            userService = container "User Service" "Управляет пользователями" "Java, Spring Boot"

            // Базы данных для каждого сервиса, обеспечивающие изоляцию данных
            // База данных проектов
            projectDatabase = container "Project Database" "Хранит информацию о проектах" "PostgreSQL"
            // База данных задач
            taskDatabase = container "Task Database" "Хранит информацию о задачах" "PostgreSQL"
            // База данных пользователей
            userDatabase = container "User Database" "Хранит информацию о пользователях" "PostgreSQL"
        }

        // Определяем взаимодействия между элементами системы

        // Пользователь взаимодействует с сервисом проектов для управления проектами
        user -> projectService "Использует для управления проектами" "HTTPS"
        // Пользователь взаимодействует с сервисом задач для управления задачами
        user -> taskService "Использует для управления задачами" "HTTPS"
        // Пользователь взаимодействует с сервисом пользователей для управления аккаунтом
        user -> userService "Использует для управления аккаунтом" "HTTPS"

        // Каждый сервис взаимодействует со своей базой данных
        projectService -> projectDatabase "Читает и записывает данные о проектах" "JDBC"
        taskService -> taskDatabase "Читает и записывает данные о задачах" "JDBC"
        userService -> userDatabase "Читает и записывает данные о пользователях" "JDBC"

        // Межсервисные коммуникации для получения необходимой информации

        // Сервис задач обращается к сервису проектов для получения информации о проектах
        taskService -> projectService "Получает данные проектов" "REST API"
        // Сервис задач обращается к сервису пользователей для получения информации о пользователях
        taskService -> userService "Получает данные пользователей" "REST API"
    }

    // Определяем представления (диаграммы) системы
    views {
        // Диаграмма контекста системы
        systemContext projectManagementSystem "SystemContext" {
            include *
            autolayout lr
        }

        // Диаграмма контейнеров системы
        container projectManagementSystem "Containers" {
            include *
            autolayout lr
        }

        // Динамическая диаграмма для иллюстрации архитектурно значимого варианта использования
        dynamic projectManagementSystem "CreateProjectAndTask" "Процесс создания проекта и задачи" {
            // Последовательность действий при создании пользователя
            user -> userService "POST /users"
            userService -> userDatabase "Создание пользователя"
            userService -> user "Подтверждение создания"

            // Последовательность действий при создании проекта
            user -> projectService "POST /projects"
            projectService -> projectDatabase "Создание проекта"
            projectService -> user "Детали проекта"

            // Последовательность действий при создании задачи
            user -> taskService "POST /tasks"
            taskService -> taskDatabase "Создание задачи"

            // Сервис задач запрашивает данные проекта у сервиса проектов
            taskService -> projectService "GET /projects/{id}"
            projectService -> projectDatabase "Получение проекта"

            // Сервис задач запрашивает данные пользователя у сервиса пользователей
            taskService -> userService "GET /users/{id}"
            userService -> userDatabase "Получение пользователя"

            // Сервис задач возвращает детали задачи пользователю
            taskService -> user "Детали задачи"
        }

        // Определяем стили для элементов диаграмм
        styles {
            // Стиль для персоны (пользователя)
            element "Person" {
                shape person
                background #08427b
                color      #ffffff
            }
            // Стиль для контейнеров (сервисов)
            element "Container" {
                shape box
                background #438dd5
                color      #ffffff
            }
            // Стиль для контейнеров-баз данных
            element "Container Database" {
                shape cylinder
                background #85bbf0
                color      #000000
            }
        }
    }
}
