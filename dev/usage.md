# Usage Guide

## Command Line Interface

The job scheduler provides a Typer-based CLI for easy job management. All commands support a `--port` option to specify a different scheduler port (default: 18765).

### Adding Jobs

Add a new job to the scheduler:
```bash
js add <job_key> <executable_path> <env_path> <interval_seconds>
```

Example:
```bash
js add backup_script ./scripts/backup.py ./envs/backup.env 86400  # Run daily
```

### Listing Jobs

View all scheduled jobs:
```bash
js list
```

Output includes:
- Job key
- Current status
- Last run time
- Next scheduled run
- Retry count

### Checking Job Status

Get detailed status of a specific job:
```bash
js status <job_key>
```

Shows:
- Current status
- Last execution time
- Next scheduled run
- Retry count
- Last error message (if any)

### Deleting Jobs

Remove a job from the scheduler:
```bash
js delete <job_key>
```

## API Usage

The scheduler provides a REST API for programmatic access:

### Check Scheduler Status
```bash
curl http://localhost:18765/
```

### List All Jobs
```bash
curl http://localhost:18765/jobs
```

### Get Job Status
```bash
curl http://localhost:18765/jobs/<job_key>
```

### Delete a Job
```bash
curl -X DELETE http://localhost:18765/jobs/<job_key>
```

### Add a New Job
```bash
curl -X POST http://localhost:18765/add_job \
    -H "Content-Type: application/json" \
    -d '{
        "job_key": "backup_script",
        "path_to_executable": "/path/to/script.py",
        "path_to_env": "/path/to/env.env",
        "schedule_interval_seconds": 86400
    }'
```

## Monitoring

### Job Logs

Job execution logs are stored in MongoDB. Each job execution record includes:
- Start and end times
- Exit code
- Standard output
- Standard error
- Environment variables used

### Error Notifications

The scheduler sends Telegram notifications for:
- Job failures
- Scheduler errors
- MongoDB connection issues

### System Logs

The scheduler logs to:
- System logs (via launchd)
- Local log files (configured in plist)
- MongoDB (job execution details)

## Best Practices

1. **Job Scripts**:
   - Use the shared virtual environment at `$DEV_ENV_PATH`
   - Include proper error handling
   - Log to stdout/stderr for capture
   - Use environment variables for configuration

2. **Scheduling**:
   - Choose appropriate intervals
   - Consider job duration when scheduling
   - Use unique job keys
   - Monitor retry counts

3. **Monitoring**:
   - Regularly check job status with `js list`
   - Monitor error notifications
   - Review execution logs
   - Watch for increasing retry counts

4. **Maintenance**:
   - Keep job scripts updated
   - Monitor disk space for logs
   - Review and clean up old jobs with `js delete`
   - Update environment files as needed 