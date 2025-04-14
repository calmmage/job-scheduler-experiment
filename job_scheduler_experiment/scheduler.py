import asyncio
import os
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import uvicorn
from aiogram import Bot
from aiogram.exceptions import TelegramAPIError
from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel, Field, field_validator
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
    schedule_interval_seconds: int
    job_key: str


class AddJobResponse(BaseModel):
    message: str
    job_key: str


# --- End FastAPI Models ---


class JobConfig(BaseModel):
    path_to_executable: Path
    path_to_env: Path
    last_run_timestamp: Optional[datetime] = None
    schedule_interval_seconds: int  # Changed from schedule_period: str
    job_key: str = Field(..., unique=True)

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


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # MongoDB settings
    mongodb_uri: str = Field(..., env="MONGODB_URI")
    mongodb_db_name: str = Field(..., env="MONGODB_DB_NAME")

    # Telegram settings
    telegram_bot_token: str = Field(..., env="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: str = Field(..., env="TELEGRAM_CHAT_ID")


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

        # Find the python executable within the specified venv
        python_executable = (
            job.path_to_env.parent / ".venv/bin/python3"
        )  # Assuming standard venv structure
        if not python_executable.exists():
            python_executable = (
                job.path_to_env.parent / "bin/python3"
            )  # Try another common structure
        if not python_executable.exists():
            raise FileNotFoundError(
                f"Python executable not found in expected venv paths near {job.path_to_env}"
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

        # Validate using the Pydantic model itself
        job_config = JobConfig(**job_data)

        # Prepare document for MongoDB (Pydantic handles datetime serialization)
        job_doc = job_config.model_dump(mode="json")
        # Convert Path objects back to strings for MongoDB
        job_doc["path_to_executable"] = str(job_doc["path_to_executable"])
        job_doc["path_to_env"] = str(job_doc["path_to_env"])

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
                        if (
                            last_run + timedelta(seconds=job.schedule_interval_seconds)
                            <= now
                        ):
                            is_due = True

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
    global jobs_collection_global
    try:
        # Load settings
        settings = Settings()

        # Initialize connections
        bot = init_telegram_bot(settings)
        mongo_client = init_mongodb(settings)

        db = mongo_client[settings.mongodb_db_name]

        # Initialize collections
        jobs_collection = db.jobs
        logs_collection = db.job_logs
        jobs_collection_global = jobs_collection  # Assign to global

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
        config = uvicorn.Config(app=app, host="0.0.0.0", port=8000, log_level="info")
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
        try:
            settings = Settings()  # Reload settings in case they weren't loaded
            bot = init_telegram_bot(settings)
            await notify_telegram(
                bot, settings.telegram_chat_id, f"Scheduler CRASHED: {e}"
            )
            await bot.session.close()  # Close bot session gracefully
        except Exception as notify_err:
            logger.error(f"Failed to send critical error notification: {notify_err}")
        sys.exit(1)


if __name__ == "__main__":
    # Ensure graceful shutdown on SIGINT/SIGTERM
    loop = asyncio.get_event_loop()
    try:
        # Start the main async function
        # asyncio.run(main()) handles loop creation and closing
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user.")
    finally:
        # Cleanup tasks etc. if needed
        # loop.run_until_complete(loop.shutdown_asyncgens()) # Optional cleanup
        logger.info("Scheduler shut down gracefully.")
