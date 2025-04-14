# Job Scheduler Tasks

## Core Features
- Create main scheduler script that will be run by plist
    - Initialize connections
        - Telegram bot (crash if fails)
        - MongoDB (retry every 20min if fails)
    - Implement job execution
        - Run jobs based on schedule
        - Handle crashes with Telegram notifications

## Storage (MongoDB)
- Jobs collection
    - path_to_executable (str)
    - path_to_env (str)
    - last_run_timestamp (datetime)
    - schedule_period (str)
- Job logs collection
    - timestamp (datetime)
    - job_key (str)
    - status (str)
    - error_message (str, optional)

## Job Management
- Design job configuration format
    - Support cron-like scheduling
    - Include job metadata (name, description, etc.)
    - Add notification settings

## Notifications
- Add Telegram bot integration
    - Create bot configuration
        - Bot token
        - Chat ID
    - Implement notification types
        - Job failures
        - Missed schedules
        - Job completion
    - Add message formatting
        - Job name
        - Error details
        - Timestamp
        - Stack trace for errors 