- [x] ~~"Set up MongoDB connection."~~
- [x] ~~"Define Job data structure in Pydantic (compatible with `job_manager_client.py`: `job_key`, `path_to_executable`, `path_to_env`, `schedule_interval_seconds`, `cron_schedule`, `python_executable_path`, `status`, `last_run`, `next_run`, `retry_count`, `error_message`)."~~
- [x] ~~"Implement FastAPI endpoints (`POST /add_job`, `GET /jobs`, `GET /jobs/{job_key}`, `DELETE /jobs/{job_key}`) to manage job definitions in MongoDB."~~
- "Set up basic APScheduler instance."
- "Implement logic to load jobs from MongoDB into APScheduler on startup and manage them via API calls."
- "Implement job execution wrapper to run the script specified by `path_to_executable` (using `python_executable_path` if provided) with environment from `path_to_env`."
- "Store job execution logs (status, stdout, stderr, timestamps) in MongoDB."
- "Update job status fields (`status`, `last_run`, `next_run`, `retry_count`, `error_message`) in MongoDB after execution."
- "Set up Telegram bot connection."
- "Implement Telegram notification on job failure." 