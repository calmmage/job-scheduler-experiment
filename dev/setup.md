# Setup Guide

## Prerequisites

- macOS (tested on macOS 13+)
- Poetry (Python package manager)
- MongoDB (local or remote)
- Telegram Bot Token (for notifications)
- External virtual environment at `$DEV_ENV_PATH/.venv`

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd job-scheduler-experiment
```

2. Install dependencies using Poetry:
```bash
poetry install
```

3. Add aliases to your `~/.aliases`:
```bash
# Job Scheduler Aliases
alias js='typer $DEV_ENV_PATH/.venv/tools/job_scheduler/job_manager_client.py run'
```

## Configuration

1. Create scheduler environment file (`scheduler.env`):
```bash
# MongoDB Configuration
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB=job_scheduler

# Telegram Configuration
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# Scheduler Configuration
API_PORT=18765
LOG_LEVEL=INFO

# Python Environment
PYTHON_PATH=$DEV_ENV_PATH/.venv/bin/python
```

2. Create job environment file (e.g., `sample_job.env`):
```bash
# Job-specific environment variables
JOB_NAME=hello_world
JOB_INTERVAL=3600

# Use the same Python environment
PYTHON_PATH=$DEV_ENV_PATH/.venv/bin/python
```

## Setting Up the Scheduler

1. Install the scheduler as a launchd service:
```bash
poetry run python setup_plist_job.py \
    --script-path scheduler.py \
    --env-path scheduler.env \
    --keep-alive \
    --python-path $DEV_ENV_PATH/.venv/bin/python
```

2. Verify the service is running:
```bash
launchctl list | grep job-scheduler
```

## Adding a Sample Job

1. Create a sample job script (`sample_jobs/hello_world.py`):
```python
#!/usr/bin/env python3
import os
from datetime import datetime

def main():
    print(f"Hello from job scheduler! Time: {datetime.now()}")
    print(f"Environment: {os.environ.get('JOB_NAME')}")
    print(f"Python Path: {os.environ.get('PYTHON_PATH')}")

if __name__ == "__main__":
    main()
```

2. Add the job to the scheduler:
```bash
js add hello_world ./sample_jobs/hello_world.py ./sample_jobs/hello_world.env 3600
```

## Managing Jobs

Use the `js` alias for all job management commands:

1. List all jobs:
```bash
js list
```

2. Check job status:
```bash
js status <job_key>
```

3. Delete a job:
```bash
js delete <job_key>
```

4. Add a new job:
```bash
js add <job_key> <executable_path> <env_path> <interval_seconds>
```

All commands support a `--port` option to specify a different scheduler port (default: 18765).

## Verifying Setup

1. Check scheduler status:
```bash
js list
```

2. Monitor job execution:
- Check MongoDB for job logs
- Watch Telegram for notifications
- View scheduler logs in the configured directory

## Troubleshooting

1. If the scheduler won't start:
- Check launchd logs: `log show --predicate 'subsystem == "com.apple.xpc.launchd"'`
- Verify environment files are correctly formatted
- Check MongoDB connection
- Verify Python path in environment files

2. If jobs aren't executing:
- Verify job paths are correct
- Check job environment files
- Monitor scheduler logs for errors
- Ensure Python path is correctly set

3. If notifications aren't working:
- Verify Telegram bot token and chat ID
- Check network connectivity
- Monitor scheduler logs for notification errors 