# Job Scheduler Experiment

A simple job scheduler that runs Python scripts on a schedule using launchd.

## Features

- [x] Setup script to create launchd plist
- [x] Scheduler functionality
- [x] MongoDB connection
- [x] Telegram notifications
- [x] FastAPI server
- [x] `/add_job` endpoint
- [x] Job management endpoints (list/status/delete)
- [x] Job execution history and retry tracking


2. Create environment files:
- `scheduler.env` for scheduler configuration
- `sample_job.env` for sample job variables

3. Set up the scheduler as a launchd job:
```bash
python setup_plist_job.py --script-path scheduler.py --env-path scheduler.env --keep-alive
```

4. Add a sample job:
```bash
python add_job_client.py hello_world ./sample_jobs/hello_world.py ./sample_jobs/hello_world.env 3600
```

## Managing Jobs

Use the `job_manager_client.py` script to manage jobs:

1. List all jobs:
```bash
python job_manager_client.py list
```

2. Check job status:
```bash
python job_manager_client.py status <job_key>
```

3. Delete a job:
```bash
python job_manager_client.py delete <job_key>
```

## API Endpoints

- `GET /` - Check if scheduler is running
- `GET /jobs` - List all jobs
- `GET /jobs/{job_key}` - Get status of a specific job
- `DELETE /jobs/{job_key}` - Delete a job
- `POST /add_job` - Add a new job

## Monitoring

- Check MongoDB for job execution history
- Watch Telegram for failure notifications
- View scheduler logs in the configured log directory

## Future Improvements

- Web interface for job management
- Job execution timeouts
- Job dependencies
- Manual job triggering
- Job pause/resume functionality