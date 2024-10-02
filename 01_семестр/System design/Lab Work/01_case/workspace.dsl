workspace {
    model {
        user = person "User" "A user of the project management system"
        
        projectManagementSystem = softwareSystem "Project Management System" "Allows users to manage projects and tasks" {
            webApplication = container "Web Application" "Provides project management functionality via web interface" "Java, Spring Boot"
            apiApplication = container "API Application" "Provides project management functionality via RESTful API" "Java, Spring Boot"
            database = container "Database" "Stores user, project, and task information" "PostgreSQL" {
                component projectEntity "Project Entity" "Stores project information" "JPA Entity"
                component taskEntity "Task Entity" "Stores task information" "JPA Entity"
                component userEntity "User Entity" "Stores user information" "JPA Entity"
            }
        }

        user -> webApplication "Uses" "HTTPS"
        user -> apiApplication "Uses" "HTTPS"
        webApplication -> database "Reads from and writes to" "JDBC"
        apiApplication -> database "Reads from and writes to" "JDBC"
    }

    views {
        systemContext projectManagementSystem "SystemContext" {
            include *
            autolayout lr
        }

        container projectManagementSystem "Containers" {
            include *
            autolayout lr
        }

        component database "DatabaseComponents" {
            include *
            autolayout lr
        }

        dynamic projectManagementSystem "CreateProjectAndTask" "Illustrates how a user creates a new project and adds a task" {
            user -> apiApplication "POST /api/projects"
            apiApplication -> database "Insert new project"
            apiApplication -> user "Return project details"
            user -> apiApplication "POST /api/projects/{projectId}/tasks"
            apiApplication -> database "Insert new task"
            apiApplication -> user "Return task details"
            user -> apiApplication "PUT /api/tasks/{taskId}/assignee"
            apiApplication -> database "Update task assignee"
            apiApplication -> user "Return updated task details"
        }
        
        styles {
            element "Person" {
                shape person
            }
            element "Container" {
                background #438dd5
                color #ffffff
            }
            element "Database" {
                shape cylinder
            }
            element "Component" {
                shape Component
            }
        }
    }
}