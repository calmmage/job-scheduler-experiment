# Job Scheduler

## What we have

- [x] [setup_plist_job.py](../job_scheduler_experiment/local_plist/setup_plist_job.py): Script to set up launchd job (KeepAlive is now optional via `--keep-alive`).
- [x] [scheduler.py](../job_scheduler_experiment/scheduler.py):
  - [x] Runs continuously, checks for due jobs, executes them asynchronously.
  - [x] Connects to MongoDB for job storage and logging.
  - [x] Notifies via Telegram on errors.
  - [x] Includes FastAPI server (`uvicorn`) running on port 8000.
  - [x] Provides `/add_job` endpoint.
    - Uses `AddJobRequest` and `AddJobResponse` Pydantic models.
- [x] [add_job_client.py](../job_scheduler_experiment/add_job_client.py): CLI utility to add jobs via the scheduler's API.


## What is the plan?

- My Custom Scheduler: [scheduler.py](../job_scheduler_experiment/scheduler.py)

- env for the scheduler

## Scheduler features

- telegram bot connection (from env file)

- mongodb connection (from env file)
  - notify via telegram if not available

## How to set up scheduler first time

- use setup_plist_job.py to set up the job
  - `python job_scheduler_experiment/local_plist/setup_plist_job.py path/to/scheduler.py --env-file path/to/.env --keep-alive` (if you want it to restart automatically)

## How to add new jobs to the scheduler

- Option 1: make it uvicorn fastapi app **(Implemented)**
  - [x] Write a script to add task to the scheduler (same interface: path, env) -> `add_job_client.py`
  - [ ] Add an alias for easy access
- Option 2: write directly to the database