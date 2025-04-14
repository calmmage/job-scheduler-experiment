# Job Scheduler

## What we have

- 1. [setup_plist_job.py](../job_scheduler_experiment/local_plist/setup_plist_job.py)

## What is the plan?

- My Custom Scheduler: [scheduler.py](../job_scheduler_experiment/scheduler.py)

- env for the scheduler

## Scheduler features

- telegram bot connection (from env file)

- mongodb connection (from env file)
  - notify via telegram if not available

## How to set up scheduler first time

- use setup_plist_job.py to set up the job
  - runp setup_plist_job.py path_to_scheduler.py --env-file path_to_env_file

## How to add new jobs to the scheduler

- Option 1: make it uvicorn fastapi app
  - Write a script to add task to the scheduler (same interface: path, env)
  - Add an alias for easy access 
- Option 2: write directly to the database