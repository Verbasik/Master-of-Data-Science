// ./mongo-init-scripts/init.js
db.createUser({
  user: 'mongoadmin',
  pwd: 'mongopass',
  roles: [
      {
          role: 'readWrite',
          db: 'task_service'
      },
      {
          role: 'dbAdmin',
          db: 'task_service'
      }
  ]
});

db = db.getSiblingDB('task_service');

// Create collections
db.createCollection('tasks');

// Create indexes
db.tasks.createIndex({ "project_id": 1 });
db.tasks.createIndex({ "creator_id": 1 });
db.tasks.createIndex({ "assignee_id": 1 });
db.tasks.createIndex({ "status": 1 });
db.tasks.createIndex({ "priority": 1 });
db.tasks.createIndex({ 
  "project_id": 1, 
  "status": 1, 
  "priority": -1 
});
db.tasks.createIndex({ "created_at": -1 });
db.tasks.createIndex({ "tags": 1 });