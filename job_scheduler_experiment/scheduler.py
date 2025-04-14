import asyncio
import os
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, List

import uvicorn
from aiogram import Bot
from aiogram.exceptions import TelegramAPIError
from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel, Field, field_validator, model_validator
from croniter import croniter
from pydantic_settings import BaseSettings, SettingsConfigDict
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from pymongo.server_api import ServerApi


# --- FastAPI Models ---
# Note: These models were defined inline in the previous version but are missing.
# Adding them back based on the add_job_endpoint signature and previous context.
class AddJobRequest(BaseModel):
    path_to_executable: str
    path_to_env: str
    schedule_interval_seconds: Optional[int] = None
    cron_schedule: Optional[str] = None
    job_key: str
    python_executable_path: Optional[str] = None  # Added field

    @model_validator(mode="before")
    @classmethod
    def check_schedule_provided(cls, values):
        interval = values.get("schedule_interval_seconds")
        cron = values.get("cron_schedule")
        if (interval is None and cron is None) or (
            interval is not None and cron is not None
        ):
            raise ValueError(
                "Exactly one of 'schedule_interval_seconds' or 'cron_schedule' must be provided."
            )
        if interval is not None and interval <= 0:
            raise ValueError("'schedule_interval_seconds' must be positive.")
        if cron is not None and not croniter.is_valid(cron):
            raise ValueError(f"Invalid cron schedule format: {cron}")
        return values


class AddJobResponse(BaseModel):
    message: str
    job_key: str


class JobStatus(BaseModel):
    job_key: str
    last_run: Optional[datetime]
    next_run: Optional[datetime]
    status: str  # "pending", "running", "success", "failure" - reflecting log status
    retry_count: int = 0  # Placeholder - retry logic not implemented
    error_message: Optional[str]


class JobListResponse(BaseModel):
    jobs: List[JobStatus]


class DeleteJobResponse(BaseModel):
    message: str
    deleted_job_key: str


# --- End FastAPI Models ---


class JobConfig(BaseModel):
    path_to_executable: Path
    path_to_env: Path
    last_run_timestamp: Optional[datetime] = None
    schedule_interval_seconds: Optional[int] = None
    cron_schedule: Optional[str] = None
    job_key: str = Field(...)
    python_executable_path: Optional[Path] = None  # Added field

    @model_validator(mode="before")
    @classmethod
    def check_schedule_stored(cls, values):
        interval = values.get("schedule_interval_seconds")
        cron = values.get("cron_schedule")
        if interval is None and cron is None:
            raise ValueError(
                "Internal error: JobConfig loaded without a schedule type."
            )
        if interval is not None and cron is not None:
            raise ValueError(
                "Internal error: JobConfig loaded with both schedule types."
            )
        if cron is not None and not croniter.is_valid(cron):
            raise ValueError(
                f"Internal error: Invalid cron schedule format stored: {cron}"
            )
        return values

    @field_validator("path_to_executable")
    def executable_must_exist(cls, v):
        if not v.exists() or not v.is_file():
            raise ValueError(f"Executable path does not exist or is not a file: {v}")
        # Basic check for python script, could be expanded
        if v.suffix != ".py":
            logger.warning(
                f"Executable {v} is not a .py file. Ensure it is executable."
            )
        return v

    @field_validator("path_to_env")
    def env_must_exist(cls, v):
        if not v.exists() or not v.is_file():
            # Allow non-existent env file path but log a warning?
            # For now, let's enforce it exists for simplicity during add.
            raise ValueError(
                f"Environment file path does not exist or is not a file: {v}"
            )
        return v

    @field_validator("python_executable_path")
    def python_executable_must_exist(cls, v):
        if v is not None:
            if not v.exists() or not v.is_file():
                raise ValueError(
                    f"Specified Python executable path does not exist or is not a file: {v}"
                )
            # Basic check if it looks like a python executable - could be enhanced
            if "python" not in v.name.lower():
                logger.warning(
                    f"Specified python executable path {v} doesn't obviously look like a Python executable."
                )
        return v


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # API settings
    api_port: int = Field(default=18765, alias="API_PORT")  # Use alias

    # MongoDB settings
    mongodb_uri: str = Field(..., alias="MONGODB_URI")
    mongodb_db_name: str = Field(..., alias="MONGODB_DB_NAME")

    # Telegram settings
    telegram_bot_token: str = Field(..., alias="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: str = Field(..., alias="TELEGRAM_CHAT_ID")


def init_telegram_bot(settings: Settings) -> Bot:
    """Initialize Telegram bot. Crashes if fails."""
    try:
        bot = Bot(token=settings.telegram_bot_token)
        return bot
    except TelegramAPIError as e:
        logger.error(f"Failed to initialize Telegram bot: {e}")
        raise


def init_mongodb(settings: Settings) -> MongoClient:
    """Initialize MongoDB connection. Retries every 20 minutes if fails."""
    while True:
        try:
            # Create a new client and connect to the server
            client = MongoClient(
                settings.mongodb_uri,
                server_api=ServerApi("1"),
                serverSelectionTimeoutMS=5000,  # 5 seconds timeout
                connectTimeoutMS=10000,  # 10 seconds connection timeout
                socketTimeoutMS=45000,  # 45 seconds socket timeout
            )

            # Test the connection
            # The ismaster command is cheap and does not require auth.
            client.admin.command("ping")
            logger.info("Successfully connected to MongoDB")
            return client

        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            logger.info("Retrying in 20 minutes...")
            # In a real scenario, consider exponential backoff or circuit breaker
            time.sleep(20 * 60)  # 20 minutes


async def notify_telegram(bot: Bot, chat_id: str, message: str):
    """Send notification to Telegram."""
    try:
        await bot.send_message(chat_id=chat_id, text=message)
    except TelegramAPIError as e:
        logger.error(f"Failed to send Telegram notification: {e}")


async def run_job(
    job: JobConfig, logs_collection, jobs_collection, bot: Bot, settings: Settings
):
    """Runs a single job asynchronously and logs the result."""
    start_time = datetime.now(timezone.utc)
    log_entry = {
        "job_key": job.job_key,
        "timestamp": start_time,
        "status": "running",
        "stdout": "",
        "stderr": "",
    }
    log_result = await asyncio.to_thread(logs_collection.insert_one, log_entry)
    log_id = log_result.inserted_id

    try:
        logger.info(f"Running job: {job.job_key} ({job.path_to_executable})")

        # Prepare environment variables
        job_env = os.environ.copy()
        if job.path_to_env.exists():
            # Basic .env file parsing - consider using python-dotenv for robustness
            with open(job.path_to_env) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        job_env[key.strip()] = value.strip()

        # Determine the Python executable to use
        if job.python_executable_path:
            # Use the path specified in the job config (already validated by Pydantic)
            python_executable = job.python_executable_path
            logger.info(f"Using job-specific python executable: {python_executable}")
        else:
            # Fall back to STABLE_VENV_PATH
            stable_venv_path_str = os.environ.get("STABLE_VENV_PATH")
            if not stable_venv_path_str:
                raise OSError(
                    "STABLE_VENV_PATH environment variable is not set and no job-specific python executable provided."
                )

            stable_venv_path = Path(stable_venv_path_str)
            python_executable = stable_venv_path / "bin/python3"
            logger.info(
                f"Using STABLE_VENV_PATH python executable: {python_executable}"
            )

            if not python_executable.exists() or not python_executable.is_file():
                raise FileNotFoundError(
                    f"Python executable not found or is not a file at the specified STABLE_VENV_PATH: {python_executable}"
                )

        process = await asyncio.create_subprocess_exec(
            str(python_executable),
            str(job.path_to_executable),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=job_env,
            cwd=job.path_to_executable.parent,  # Run in the script's directory
        )

        stdout, stderr = await process.communicate()
        stdout_str = stdout.decode().strip() if stdout else ""
        stderr_str = stderr.decode().strip() if stderr else ""
        end_time = datetime.now(timezone.utc)

        if process.returncode == 0:
            status = "success"
            logger.info(f"Job {job.job_key} completed successfully.")
        else:
            status = "failure"
            logger.error(
                f"Job {job.job_key} failed with return code {process.returncode}."
            )
            logger.error(f"Stderr: {stderr_str}")
            await notify_telegram(
                bot,
                settings.telegram_chat_id,
                f"Job '{job.job_key}' failed!\nReturn Code: {process.returncode}\nStderr: {stderr_str[:1000]}",  # Truncate stderr
            )

        # Update log entry
        await asyncio.to_thread(
            logs_collection.update_one,
            {"_id": log_id},
            {
                "$set": {
                    "status": status,
                    "stdout": stdout_str,
                    "stderr": stderr_str,
                    "end_timestamp": end_time,
                    "duration_seconds": (end_time - start_time).total_seconds(),
                    "return_code": process.returncode,
                }
            },
        )

        # Update last run timestamp for the job
        await asyncio.to_thread(
            jobs_collection.update_one,
            {"job_key": job.job_key},
            {
                "$set": {"last_run_timestamp": start_time}
            },  # Use start_time for consistency
        )

    except FileNotFoundError as e:
        end_time = datetime.now(timezone.utc)
        status = "failure"
        error_message = f"Job {job.job_key} failed: {e}"
        logger.error(error_message)
        await notify_telegram(bot, settings.telegram_chat_id, error_message)
        await asyncio.to_thread(
            logs_collection.update_one,
            {"_id": log_id},
            {
                "$set": {
                    "status": status,
                    "stderr": error_message,
                    "end_timestamp": end_time,
                }
            },
        )
    except Exception as e:
        end_time = datetime.now(timezone.utc)
        status = "failure"
        error_message = f"Job {job.job_key} crashed: {e}\n{traceback.format_exc()}"
        logger.error(error_message)
        await notify_telegram(
            bot, settings.telegram_chat_id, f"Job '{job.job_key}' crashed! See logs."
        )
        # Update log entry with error
        await asyncio.to_thread(
            logs_collection.update_one,
            {"_id": log_id},
            {
                "$set": {
                    "status": status,
                    "stderr": error_message,
                    "end_timestamp": end_time,
                }
            },
        )


app = FastAPI()
jobs_collection_global = None  # Global variable to hold the collection
logs_collection_global = None  # Global variable for logs collection


@app.post("/add_job", response_model=AddJobResponse)
async def add_job_endpoint(job_request: AddJobRequest):
    global jobs_collection_global
    if jobs_collection_global is None:
        raise HTTPException(
            status_code=503,
            detail="Scheduler not fully initialized, database unavailable.",
        )

    try:
        # Convert paths and validate JobConfig
        job_data = job_request.model_dump()
        job_data["path_to_executable"] = Path(job_data["path_to_executable"]).resolve()
        job_data["path_to_env"] = Path(job_data["path_to_env"]).resolve()
        if job_data.get("python_executable_path"):
            job_data["python_executable_path"] = Path(
                job_data["python_executable_path"]
            ).resolve()

        # Validate using the Pydantic model itself
        job_config = JobConfig(**job_data)

        # Prepare document for MongoDB (Pydantic handles datetime serialization)
        job_doc = job_config.model_dump(mode="json")
        # Convert Path objects back to strings for MongoDB
        job_doc["path_to_executable"] = str(job_doc["path_to_executable"])
        job_doc["path_to_env"] = str(job_doc["path_to_env"])
        if job_doc.get("python_executable_path"):
            job_doc["python_executable_path"] = str(job_doc["python_executable_path"])

        # Insert into MongoDB - this needs to be async
        try:
            # Use asyncio.to_thread for synchronous pymongo operation
            insert_result = await asyncio.to_thread(
                jobs_collection_global.insert_one, job_doc
            )
            logger.info(
                f"Successfully added job: {job_request.job_key} with id {insert_result.inserted_id}"
            )
            return AddJobResponse(
                message="Job added successfully", job_key=job_request.job_key
            )
        except Exception as e:  # Catch potential duplicate key errors etc.
            # Check if it's a duplicate key error (code 11000)
            if "E11000 duplicate key error collection" in str(e):
                logger.warning(
                    f"Attempted to add duplicate job_key: {job_request.job_key}"
                )
                raise HTTPException(
                    status_code=409,
                    detail=f"Job with key '{job_request.job_key}' already exists.",
                )
            else:
                logger.error(
                    f"Failed to insert job {job_request.job_key} into MongoDB: {e}"
                )
                raise HTTPException(
                    status_code=500, detail=f"Database error when adding job: {e}"
                )

    except ValueError as e:
        logger.error(f"Validation error adding job {job_request.job_key}: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error adding job {job_request.job_key}: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")


async def scheduler_loop(
    settings: Settings, bot: Bot, jobs_collection, logs_collection
):
    """The main scheduling loop, extracted from main."""
    logger.info("Scheduler loop started. Checking for jobs...")
    await notify_telegram(bot, settings.telegram_chat_id, "Scheduler loop started.")
    while True:
        now = datetime.now(timezone.utc)
        try:
            # Find jobs that are due
            # Use asyncio.to_thread for the blocking find operation
            due_jobs_cursor = await asyncio.to_thread(
                jobs_collection.find,
                {
                    "$or": [
                        {"last_run_timestamp": None},  # Never run before
                        {
                            "$expr": {
                                "$lte": [
                                    {
                                        "$add": [
                                            "$last_run_timestamp",
                                            {
                                                "$multiply": [
                                                    "$schedule_interval_seconds",
                                                    1000,
                                                ]
                                            },
                                        ]
                                    },
                                    now,
                                ]
                            }
                        },
                    ]
                },
            )

            # Need to iterate over the cursor potentially blocking, wrap in thread?
            # Or fetch all results into a list (might use memory for many jobs)
            due_jobs_list = await asyncio.to_thread(list, due_jobs_cursor)

            for job_doc in due_jobs_list:
                try:
                    job = JobConfig(**job_doc)
                    # Check again more accurately (DB query is approximate)
                    is_due = False
                    if job.last_run_timestamp is None:
                        is_due = True
                    else:
                        # Make last_run_timestamp timezone-aware if it's not
                        last_run = (
                            job.last_run_timestamp.replace(tzinfo=timezone.utc)
                            if job.last_run_timestamp.tzinfo is None
                            else job.last_run_timestamp
                        )
                        # Check interval
                        if job.schedule_interval_seconds is not None:
                            if (
                                last_run
                                + timedelta(seconds=job.schedule_interval_seconds)
                                <= now
                            ):
                                is_due = True
                        # Check cron
                        elif job.cron_schedule is not None:
                            # Get the next scheduled time AFTER the last run time
                            try:
                                cron = croniter(job.cron_schedule, last_run)
                                next_scheduled_run = cron.get_next(datetime)
                                if next_scheduled_run <= now:
                                    is_due = True
                            except Exception as e:
                                logger.error(
                                    f"Error calculating next cron run check for {job.job_key}: {e}"
                                )

                    if is_due:
                        # Optimistic lock: Try to update last_run_timestamp immediately
                        result = await asyncio.to_thread(
                            jobs_collection.update_one,
                            {
                                "job_key": job.job_key,
                                "last_run_timestamp": job.last_run_timestamp,
                            },
                            {"$set": {"last_run_timestamp": now}},
                        )

                        if result.modified_count == 1:
                            logger.info(f"Job {job.job_key} is due. Creating task.")
                            # Run the job in the background
                            asyncio.create_task(
                                run_job(
                                    job, logs_collection, jobs_collection, bot, settings
                                )
                            )
                        else:
                            logger.info(
                                f"Job {job.job_key} was likely picked up by another instance or run very recently. Skipping."
                            )

                except Exception as e:
                    logger.error(
                        f"Error processing job document {job_doc.get('_id', 'N/A')}: {e}\n{traceback.format_exc()}"
                    )
                    await notify_telegram(
                        bot,
                        settings.telegram_chat_id,
                        f"Error processing job document {job_doc.get('_id', 'N/A')}. See logs.",
                    )

        except Exception as loop_error:
            logger.error(
                f"Error in scheduler loop iteration: {loop_error}\n{traceback.format_exc()}"
            )
            await notify_telegram(
                bot,
                settings.telegram_chat_id,
                "Error in scheduler loop iteration. See logs.",
            )

        # Wait before checking again
        await asyncio.sleep(60)


async def main():
    global jobs_collection_global, logs_collection_global  # Add logs_collection_global
    try:
        # Load settings
        settings = Settings()  # type: ignore

        # Initialize connections
        bot = init_telegram_bot(settings)
        mongo_client = init_mongodb(settings)

        db = mongo_client[settings.mongodb_db_name]

        # Initialize collections
        jobs_collection = db.jobs
        logs_collection = db.job_logs
        jobs_collection_global = jobs_collection  # Assign to global
        logs_collection_global = logs_collection  # Assign logs collection to global

        # Ensure indexes (Run synchronously before starting async loops)
        # Using .command directly might be better if async driver is not used
        # For simplicity with pymongo, keep sync calls here before event loop starts if possible,
        # or ensure they are awaited properly if loop is already running.
        # Let's assume we run them before the main uvicorn server starts.
        logger.info("Ensuring MongoDB indexes...")
        jobs_collection.create_index("last_run_timestamp")
        jobs_collection.create_index("schedule_interval_seconds")
        jobs_collection.create_index("job_key", unique=True)
        logs_collection.create_index("timestamp")
        logs_collection.create_index("job_key")
        logs_collection.create_index("status")
        logger.info("MongoDB indexes ensured.")

        # Configure Uvicorn server
        config = uvicorn.Config(
            app=app,
            host="0.0.0.0",
            port=settings.api_port,  # Use configured port
            log_level="info",
        )
        server = uvicorn.Server(config)

        # Start the scheduler loop as a background task
        scheduler_task = asyncio.create_task(
            scheduler_loop(settings, bot, jobs_collection, logs_collection)
        )

        logger.info("Starting FastAPI server on port 8000...")
        # Start the Uvicorn server
        await server.serve()

        # Wait for the scheduler task to complete (it won't, unless cancelled)
        await scheduler_task

    except Exception as e:
        logger.error(f"Critical error in main loop: {e}\n{traceback.format_exc()}")
        # Try to notify Telegram before exiting (best effort)
        # Linter might complain here if env vars aren't set, which is expected.
        try:
            settings = Settings()  # type: ignore
            bot = init_telegram_bot(settings)
            await notify_telegram(
                bot, settings.telegram_chat_id, f"Scheduler CRASHED: {e}"
            )
            await bot.session.close()  # Close bot session gracefully
        except Exception as notify_err:
            logger.error(f"Failed to send critical error notification: {notify_err}")
        sys.exit(1)


@app.get("/")
async def root():
    return {"status": "running"}


@app.get("/jobs", response_model=JobListResponse)
async def list_jobs():
    """List all jobs with their current status."""
    if (
        jobs_collection_global is None or logs_collection_global is None
    ):  # Check both collections
        raise HTTPException(status_code=503, detail="Database unavailable.")

    jobs = []
    try:
        # Get all job configs
        job_docs_cursor = await asyncio.to_thread(jobs_collection_global.find, {})
        job_configs = await asyncio.to_thread(list, job_docs_cursor)

        # For each job, get the latest log status
        for job_doc in job_configs:
            job_key = job_doc["job_key"]
            last_run_ts = job_doc.get("last_run_timestamp")
            interval = job_doc.get("schedule_interval_seconds")

            # Find the most recent log entry for this job
            latest_log_cursor = await asyncio.to_thread(
                logs_collection_global.find, {"job_key": job_key}
            )
            # Sort by timestamp descending and get the first one
            latest_log_list = await asyncio.to_thread(
                lambda: list(latest_log_cursor.sort("timestamp", -1).limit(1))
            )
            latest_log = latest_log_list[0] if latest_log_list else None

            status = "pending"  # Default if no logs found
            error_message = None
            if latest_log:
                status = latest_log.get("status", "unknown")  # Use log status
                # Use stderr as error message if status is failure
                if status == "failure":
                    error_message = latest_log.get("stderr")
                elif (
                    latest_log.get("return_code") is not None
                    and latest_log.get("return_code") != 0
                ):
                    # Catch cases where status might be 'running' but process exited non-zero (e.g. crash during run)
                    status = "failure"
                    error_message = latest_log.get(
                        "stderr",
                        "Job exited with non-zero code but no stderr captured.",
                    )

            # Calculate next run time
            next_run = None
            if last_run_ts and interval:
                last_run_aware = (
                    last_run_ts.replace(tzinfo=timezone.utc)
                    if last_run_ts.tzinfo is None
                    else last_run_ts
                )
                if interval is not None:
                    next_run = last_run_aware + timedelta(seconds=interval)
            elif interval and not last_run_ts:
                # First run logic for interval: leave as None in status? Or calculate from now?
                # Let's leave as None for status display consistency.
                pass
            elif last_run_ts and job_doc.get("cron_schedule"):
                last_run_aware = (
                    last_run_ts.replace(tzinfo=timezone.utc)
                    if last_run_ts.tzinfo is None
                    else last_run_ts
                )
                try:
                    cron_sched = job_doc["cron_schedule"]
                    # Need a base time for croniter
                    cron = croniter(cron_sched, last_run_aware)
                    next_run = cron.get_next(datetime)
                except Exception as e:
                    logger.error(
                        f"Error calculating next cron run for {job_key} in list: {e}"
                    )
            elif interval and not last_run_ts:
                # First run logic for interval: leave as None in status? Or calculate from now?
                # Let's leave as None for status display consistency.
                pass
            elif job_doc.get("cron_schedule") and not last_run_ts:
                try:
                    # First run for cron based on 'now'
                    now_local = datetime.now(timezone.utc)  # Use current time
                    cron_sched = job_doc["cron_schedule"]
                    cron = croniter(cron_sched, now_local)
                    next_run = cron.get_next(datetime)
                except Exception as e:
                    logger.error(
                        f"Error calculating first cron run for {job_key} in list: {e}"
                    )

            job_status = JobStatus(
                job_key=job_key,
                last_run=last_run_ts,
                next_run=next_run,
                status=status,
                retry_count=0,  # Hardcoded for now
                error_message=error_message,
            )
            jobs.append(job_status)
        return JobListResponse(jobs=jobs)
    except Exception as e:
        logger.error(f"Error listing jobs: {e}\\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Error retrieving job list.")


@app.get("/jobs/{job_key}", response_model=JobStatus)
async def get_job_status(job_key: str):
    """Get status of a specific job."""
    if jobs_collection_global is None or logs_collection_global is None:
        raise HTTPException(status_code=503, detail="Database unavailable.")

    try:
        # Get job config
        job_doc = await asyncio.to_thread(
            jobs_collection_global.find_one, {"job_key": job_key}
        )

        if not job_doc:
            raise HTTPException(status_code=404, detail="Job not found")

        # Find the most recent log entry for this job
        latest_log_cursor = await asyncio.to_thread(
            logs_collection_global.find, {"job_key": job_key}
        )
        latest_log_list = await asyncio.to_thread(
            lambda: list(latest_log_cursor.sort("timestamp", -1).limit(1))
        )
        latest_log = latest_log_list[0] if latest_log_list else None

        status = "pending"  # Default if no logs found
        error_message = None
        if latest_log:
            status = latest_log.get("status", "unknown")
            if status == "failure":
                error_message = latest_log.get("stderr")
            elif (
                latest_log.get("return_code") is not None
                and latest_log.get("return_code") != 0
            ):
                status = "failure"
                error_message = latest_log.get(
                    "stderr", "Job exited with non-zero code but no stderr captured."
                )

        # Calculate next run time
        next_run = None
        last_run_ts = job_doc.get("last_run_timestamp")
        interval = job_doc.get("schedule_interval_seconds")
        if last_run_ts and interval:
            last_run_aware = (
                last_run_ts.replace(tzinfo=timezone.utc)
                if last_run_ts.tzinfo is None
                else last_run_ts
            )
            if interval is not None:
                next_run = last_run_aware + timedelta(seconds=interval)
        elif last_run_ts and job_doc.get("cron_schedule"):
            last_run_aware = (
                last_run_ts.replace(tzinfo=timezone.utc)
                if last_run_ts.tzinfo is None
                else last_run_ts
            )
            try:
                cron_sched = job_doc["cron_schedule"]
                cron = croniter(cron_sched, last_run_aware)
                next_run = cron.get_next(datetime)
            except Exception as e:
                logger.error(
                    f"Error calculating next cron run for {job_key} in status: {e}"
                )
        elif interval and not last_run_ts:
            # First run logic for interval: leave as None in status? Or calculate from now?
            # Let's leave as None for status display consistency.
            pass
        elif job_doc.get("cron_schedule") and not last_run_ts:
            try:
                # First run for cron based on 'now'
                now_local = datetime.now(timezone.utc)  # Use current time
                cron_sched = job_doc["cron_schedule"]
                cron = croniter(cron_sched, now_local)
                next_run = cron.get_next(datetime)
            except Exception as e:
                logger.error(
                    f"Error calculating first cron run for {job_key} in status: {e}"
                )

        return JobStatus(
            job_key=job_doc["job_key"],
            last_run=last_run_ts,
            next_run=next_run,
            status=status,
            retry_count=0,  # Hardcoded
            error_message=error_message,
        )
    except HTTPException:  # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(
            f"Error getting status for job {job_key}: {e}\\n{traceback.format_exc()}"
        )
        raise HTTPException(status_code=500, detail="Error retrieving job status.")


@app.delete("/jobs/{job_key}", response_model=DeleteJobResponse)
async def delete_job(job_key: str):
    """Delete a job from the scheduler."""
    if jobs_collection_global is None:
        raise HTTPException(status_code=503, detail="Database unavailable.")

    try:
        # Use asyncio.to_thread for the blocking delete_one operation
        delete_result = await asyncio.to_thread(
            jobs_collection_global.delete_one, {"job_key": job_key}
        )

        if delete_result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Job not found")

        logger.info(f"Deleted job: {job_key}")
        return DeleteJobResponse(
            message="Job deleted successfully", deleted_job_key=job_key
        )
    except HTTPException:  # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error deleting job {job_key}: {e}")
        raise HTTPException(status_code=500, detail="Error deleting job.")


if __name__ == "__main__":
    # Ensure graceful shutdown on SIGINT/SIGTERM
    loop = asyncio.get_event_loop()
    # Initialize settings once here to potentially catch config errors early
    # Linter might complain if env vars aren't set, which is expected.
    try:
        settings = Settings()  # type: ignore
    except Exception as config_e:
        logger.critical(f"Failed to load settings: {config_e}")
        sys.exit(1)

    try:
        # Pass settings to main to avoid global state issues if possible
        # Note: Current main() uses globals, refactoring needed for pure DI
        asyncio.run(main())  # main() still reads settings internally
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user.")
    except Exception as main_e:
        logger.critical(
            f"Unhandled exception in main execution: {main_e}\n{traceback.format_exc()}"
        )
        sys.exit(1)
    finally:
        logger.info("Scheduler attempting graceful shutdown.")
        # Add any explicit cleanup needed here
