import asyncio
import os
import traceback
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, cast

import uvicorn
from aiogram import Bot
from aiogram.exceptions import TelegramAPIError
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from fastapi import Depends, FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from pymongo.server_api import ServerApi
from pymongo.database import Database

from .scheduler_app import SchedulerApp

# --- Global State ---
# Using a dictionary for simple state management during lifespan
app_state: Dict[str, Any] = {}


# --- FastAPI Models ---
class AddJobRequest(BaseModel):
    job_id: str
    interval: int
    message: str


class AddJobResponse(BaseModel):
    message: str
    job_id: str


class JobStatus(BaseModel):
    job_id: str
    name: str
    command: str
    schedule: str
    last_run: Optional[str] = None
    next_run: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    status: str = "pending"


class JobListResponse(BaseModel):
    jobs: List[JobStatus]


class DeleteJobResponse(BaseModel):
    message: str
    deleted_job_id: str


# --- End FastAPI Models ---


class JobConfig(BaseModel):
    job_id: str
    name: str
    command: str
    schedule: str
    telegram_chat_id: Optional[str] = None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
    api_port: int = Field(default=18765, alias="API_PORT")
    mongodb_uri: str = Field(..., alias="MONGODB_URI")
    mongodb_db_name: str = Field(..., alias="MONGODB_DB_NAME")
    telegram_bot_token: str = Field(..., alias="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: str = Field(..., alias="TELEGRAM_CHAT_ID")
    mongodb_server_api: str = "1"


# --- Initialization Functions ---


def init_telegram_bot(settings: Settings) -> Bot:
    try:
        bot = Bot(token=settings.telegram_bot_token)
        logger.info("Telegram bot initialized.")
        return bot
    except TelegramAPIError as e:
        logger.error(f"Failed to initialize Telegram bot: {e}")
        raise


def init_mongodb(settings: Settings) -> MongoClient:
    # Reduced retry logic for startup, fail faster if DB isn't ready initially
    try:
        client = MongoClient(
            settings.mongodb_uri,
            server_api=ServerApi("1"),
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=10000,
            socketTimeoutMS=15000,  # Reduced socket timeout for init
        )
        client.admin.command("ping")
        logger.info("Successfully connected to MongoDB.")
        return client
    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        logger.error(f"Failed to connect to MongoDB on startup: {e}")
        raise


async def notify_telegram(bot: Bot, chat_id: str, message: str):
    try:
        await bot.send_message(chat_id=chat_id, text=message)
    except TelegramAPIError as e:
        logger.error(f"Failed to send Telegram notification: {e}")
    except Exception as e:
        logger.error(f"Unexpected error sending Telegram notification: {e}")


# --- Job Execution Logic ---


async def run_job_wrapper(job_id: str):
    """
    Wrapper function called by APScheduler.
    It retrieves necessary dependencies from app_state and calls the actual run_job.
    """
    logger.debug(f"Scheduler triggered run_job_wrapper for job_id: {job_id}")
    # Retrieve dependencies from global state
    jobs_collection = app_state.get("jobs_collection")
    logs_collection = app_state.get("logs_collection")
    bot = app_state.get("bot")
    settings = app_state.get("settings")

    assert jobs_collection is not None
    assert logs_collection is not None
    assert bot is not None
    assert settings is not None

    try:
        # Fetch job config from DB
        job_doc = await asyncio.to_thread(jobs_collection.find_one, {"job_id": job_id})
        if not job_doc:
            logger.error(
                f"Job {job_id} not found in database for execution. Cancelling scheduled task."
            )
            # Try to cancel the orphaned job in the scheduler
            scheduler: AsyncIOScheduler = app_state.get("scheduler")
            if scheduler:
                try:
                    scheduler.remove_job(job_id)
                    logger.info(f"Cancelled orphaned job {job_id} in APScheduler.")
                except Exception as cancel_err:
                    logger.error(
                        f"Failed to cancel orphaned job {job_id} in APScheduler: {cancel_err}"
                    )
            return

        # Convert DB doc to Pydantic model (handles Path conversion etc.)
        job_config = JobConfig(**job_doc)

        # Call the actual job execution logic
        await execute_job_script(
            job_config, logs_collection, jobs_collection, bot, settings
        )

    except Exception as e:
        logger.error(
            f"Error during run_job_wrapper for {job_id}: {e}\n{traceback.format_exc()}"
        )
        # Attempt to notify about wrapper failure
        await notify_telegram(
            bot,
            settings.telegram_chat_id,
            f"Scheduler wrapper failed for job {job_id}: {e}",
        )


async def execute_job_script(
    job: JobConfig, logs_collection, jobs_collection, bot: Bot, settings: Settings
):
    """
    The core logic to execute the job's Python script.
    (Renamed from run_job to avoid confusion with the wrapper)
    """
    start_time = datetime.now(timezone.utc)
    log_entry = {
        "job_id": job.job_id,
        "timestamp": start_time,
        "status": "running",
        "stdout": "",
        "stderr": "",
    }

    # Insert 'running' log immediately
    try:
        log_result = await asyncio.to_thread(logs_collection.insert_one, log_entry)
        log_id = log_result.inserted_id
    except Exception as log_insert_err:
        logger.error(
            f"Failed to insert initial 'running' log for {job.job_id}: {log_insert_err}"
        )
        # Attempt to notify and then exit - can't proceed without log entry
        await notify_telegram(
            bot, settings.telegram_chat_id, f"Failed to log start for job {job.job_id}"
        )
        return

    try:
        logger.info(f"Executing job script: {job.job_id} ({job.command})")

        # Prepare environment variables
        job_env = os.environ.copy()
        if job.telegram_chat_id:
            job_env["TELEGRAM_CHAT_ID"] = job.telegram_chat_id

        # Execute the script
        process = await asyncio.create_subprocess_exec(
            job.command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=job_env,
        )
        stdout, stderr = await process.communicate()
        stdout_str = stdout.decode(errors="ignore").strip() if stdout else ""
        stderr_str = stderr.decode(errors="ignore").strip() if stderr else ""
        end_time = datetime.now(timezone.utc)
        duration = (end_time - start_time).total_seconds()

        # Determine outcome
        if process.returncode == 0:
            status = "success"
            logger.info(f"Job {job.job_id} completed successfully in {duration:.2f}s.")
        else:
            status = "failure"
            error_message_short = f"Job '{job.job_id}' failed! RC: {process.returncode}, Stderr: {stderr_str[:500]}"
            logger.error(
                f"Job {job.job_id} failed. RC: {process.returncode}, Duration: {duration:.2f}s."
            )
            logger.error(f"Stderr: {stderr_str}")
            await notify_telegram(bot, settings.telegram_chat_id, error_message_short)

        # Update log entry with final status
        update_data = {
            "status": status,
            "stdout": stdout_str,
            "stderr": stderr_str,
            "end_timestamp": end_time,
            "duration_seconds": duration,
            "return_code": process.returncode,
        }
        await asyncio.to_thread(
            logs_collection.update_one, {"_id": log_id}, {"$set": update_data}
        )

        # Update last run timestamp in jobs collection
        await asyncio.to_thread(
            jobs_collection.update_one,
            {"job_id": job.job_id},
            {"$set": {"last_run_timestamp": start_time}},
        )

    except (FileNotFoundError, OSError) as env_err:  # Catch env/path setup errors
        end_time = datetime.now(timezone.utc)
        status = "failure"
        error_message = f"Job {job.job_id} setup failed: {env_err}"
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
    except Exception as exec_err:  # Catch unexpected execution errors
        end_time = datetime.now(timezone.utc)
        status = "failure"
        error_message = (
            f"Job {job.job_id} execution crashed: {exec_err}\n{traceback.format_exc()}"
        )
        logger.error(error_message)
        await notify_telegram(
            bot, settings.telegram_chat_id, f"Job '{job.job_id}' crashed! See logs."
        )
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


# --- FastAPI Lifespan Management ---


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles application startup and shutdown logic."""
    global app_state
    logger.info("Application startup...")

    # Initialize Settings, Bot, MongoDB, Scheduler
    try:
        settings = Settings()  # type: ignore
        app_state["settings"] = settings
        app_state["bot"] = init_telegram_bot(settings)
        mongo_client = init_mongodb(settings)
        app_state["mongo_client"] = mongo_client
        db = mongo_client[settings.mongodb_db_name]
        app_state["jobs_collection"] = db.jobs
        app_state["logs_collection"] = db.job_logs

        # Ensure MongoDB indexes
        logger.info("Ensuring MongoDB indexes...")
        await asyncio.to_thread(db.jobs.create_index, "job_id", unique=True)
        await asyncio.to_thread(db.jobs.create_index, "last_run_timestamp")
        await asyncio.to_thread(db.job_logs.create_index, "timestamp")
        await asyncio.to_thread(db.job_logs.create_index, "job_id")
        await asyncio.to_thread(db.job_logs.create_index, "status")
        logger.info("MongoDB indexes ensured.")

        # Initialize APScheduler
        scheduler = AsyncIOScheduler()
        app_state["scheduler"] = scheduler

        # Load and schedule existing jobs from DB
        logger.info("Loading and scheduling existing jobs from database...")
        jobs_collection = app_state["jobs_collection"]
        job_docs_cursor = await asyncio.to_thread(jobs_collection.find, {})
        existing_jobs = await asyncio.to_thread(list, job_docs_cursor)

        scheduled_count = 0
        for job_doc in existing_jobs:
            job_id = job_doc["job_id"]
            try:
                scheduler.add_job(
                    run_job_wrapper,
                    IntervalTrigger(seconds=int(job_doc["schedule"])),
                    args=[job_id],
                    id=job_id,
                    replace_existing=True,
                )
                logger.info(f"Scheduled existing job: {job_id}")
                scheduled_count += 1
            except Exception as schedule_err:
                logger.error(
                    f"Failed to schedule existing job {job_id}: {schedule_err}"
                )

        logger.info(
            f"Finished scheduling existing jobs. Total scheduled: {scheduled_count}"
        )

        # Start the scheduler
        scheduler.start()
        logger.info("APScheduler started.")
        await notify_telegram(
            app_state["bot"],
            app_state["settings"].telegram_chat_id,
            "Scheduler Service Started",
        )

        yield  # Application runs here

    except Exception as startup_err:
        logger.critical(
            f"Application startup failed: {startup_err}\n{traceback.format_exc()}"
        )
        # Attempt cleanup if possible
        if "bot" in app_state and app_state["bot"].session:
            await app_state["bot"].session.close()
        if "scheduler" in app_state and app_state["scheduler"].running:
            app_state["scheduler"].shutdown()
        if "mongo_client" in app_state:
            app_state["mongo_client"].close()
        # Re-raise to prevent FastAPI from starting improperly
        raise startup_err
    finally:
        # --- Shutdown Logic ---
        logger.info("Application shutdown...")
        scheduler = app_state.get("scheduler")
        if scheduler and scheduler.running:
            logger.info("Stopping APScheduler...")
            try:
                scheduler.shutdown()
                logger.info("APScheduler stopped.")
            except Exception as e:
                logger.error(f"Error stopping APScheduler: {e}")

        bot = app_state.get("bot")
        if bot and bot.session:
            logger.info("Closing Telegram bot session...")
            try:
                await bot.session.close()
                logger.info("Telegram bot session closed.")
            except Exception as e:
                logger.error(f"Error closing bot session: {e}")

        mongo_client = app_state.get("mongo_client")
        if mongo_client:
            logger.info("Closing MongoDB connection...")
            mongo_client.close()
            logger.info("MongoDB connection closed.")

        # Attempt to notify about shutdown (best effort)
        if bot and settings:
            try:
                # Need to create a temporary session to send final message
                async with Bot(token=settings.telegram_bot_token) as temp_bot:
                    await notify_telegram(
                        temp_bot, settings.telegram_chat_id, "Scheduler Service Stopped"
                    )
            except Exception as final_notify_err:
                logger.error(
                    f"Failed to send final shutdown notification: {final_notify_err}"
                )

        logger.info("Application shutdown complete.")
        app_state.clear()


# --- FastAPI App and Endpoints ---


class AppState:
    def __init__(self):
        self.scheduler: AsyncIOScheduler = AsyncIOScheduler()
        self.scheduler_app: Optional[SchedulerApp] = None
        self.bot: Optional[Bot] = None
        self.db: Optional[Database] = None


app = FastAPI(lifespan=lifespan)
app.state = AppState()


# Dependency to get scheduler instance
async def get_scheduler() -> AsyncIOScheduler:
    scheduler = app.state.scheduler
    if scheduler is None:
        raise RuntimeError("Scheduler not initialized")
    return cast(AsyncIOScheduler, scheduler)


# Dependency to get settings
def get_settings() -> Settings:
    settings = app_state.get("settings")
    if not settings:
        # This ideally shouldn't happen if lifespan management is correct
        raise HTTPException(status_code=503, detail="Settings not available.")
    return settings


# Dependency for DB collections (can be split if needed)
def get_jobs_collection():
    collection = app_state.get("jobs_collection")
    if not collection:
        raise HTTPException(status_code=503, detail="Jobs DB collection not available.")
    return collection


def get_logs_collection():
    collection = app_state.get("logs_collection")
    if not collection:
        raise HTTPException(status_code=503, detail="Logs DB collection not available.")
    return collection


@app.post("/jobs", response_model=AddJobResponse)
async def add_job_endpoint(
    job_request: AddJobRequest,
    scheduler: AsyncIOScheduler = Depends(get_scheduler),
    jobs_collection=Depends(get_jobs_collection),
    settings: Settings = Depends(get_settings),
):
    """Adds a job to the database and schedules it."""
    job_id = job_request.job_id
    logger.info(f"Received request to add job: {job_id}")

    # Check if job already exists in scheduler
    if scheduler.get_job(job_id):
        logger.warning(f"Job id '{job_id}' already exists in the scheduler.")
        raise HTTPException(
            status_code=409, detail=f"Job with id '{job_id}' is already scheduled."
        )

    try:
        # --- Database Operation ---
        # Convert paths and validate JobConfig
        job_data = job_request.model_dump()
        job_data["command"] = job_data["command"]
        job_data["schedule"] = job_data["schedule"]
        job_data["telegram_chat_id"] = job_data.get("telegram_chat_id")

        # Validate model instance (Pydantic)
        job_config = JobConfig(**job_data)

        # Prepare doc for MongoDB
        job_doc = job_config.model_dump(mode="json")
        job_doc["command"] = job_doc["command"]
        job_doc["schedule"] = job_doc["schedule"]
        if job_doc.get("telegram_chat_id"):
            job_doc["telegram_chat_id"] = job_doc["telegram_chat_id"]

        # Insert into MongoDB
        try:
            await asyncio.to_thread(jobs_collection.insert_one, job_doc)
            logger.info(f"Successfully added job config to DB: {job_id}")
        except Exception as e:
            if "E11000 duplicate key error collection" in str(e):
                logger.warning(f"Attempted to add duplicate job_id to DB: {job_id}")
                raise HTTPException(
                    status_code=409,
                    detail=f"Job config with id '{job_id}' already exists in DB.",
                )
            else:
                logger.error(f"Failed to insert job {job_id} into MongoDB: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Database error when adding job config: {e}",
                )

        # --- Scheduling Operation ---
        try:
            scheduler.add_job(
                run_job_wrapper,
                IntervalTrigger(seconds=int(job_config.schedule)),
                args=[job_id],
                id=job_id,
                replace_existing=True,
            )
            logger.info(f"Successfully scheduled job: {job_id}")
            return AddJobResponse(
                message="Job added and scheduled successfully", job_id=job_id
            )

        except Exception as schedule_err:
            logger.error(
                f"Failed to schedule job {job_id} after DB insert: {schedule_err}"
            )
            # Attempt to clean up the DB entry if scheduling fails
            try:
                await asyncio.to_thread(jobs_collection.delete_one, {"job_id": job_id})
                logger.warning(
                    f"Removed DB entry for {job_id} due to scheduling failure."
                )
            except Exception as cleanup_err:
                logger.error(
                    f"Failed to cleanup DB entry for {job_id} after scheduling error: {cleanup_err}"
                )
            raise HTTPException(
                status_code=500, detail=f"Failed to schedule job: {schedule_err}"
            )

    except ValueError as e:  # Catches Pydantic validation errors
        logger.error(f"Validation error adding job {job_id}: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:  # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(
            f"Unexpected error in add_job_endpoint for {job_id}: {e}\n{traceback.format_exc()}"
        )
        raise HTTPException(
            status_code=500, detail="An unexpected server error occurred."
        )


@app.get("/jobs", response_model=JobListResponse)
async def list_jobs(
    jobs_collection=Depends(get_jobs_collection),
    logs_collection=Depends(get_logs_collection),
):
    """Lists jobs based on database entries and latest log status."""
    logger.debug("Request received for listing jobs.")
    jobs = []
    try:
        # Get all job configs from DB
        job_docs_cursor = await asyncio.to_thread(jobs_collection.find, {})
        job_configs = await asyncio.to_thread(list, job_docs_cursor)
        logger.debug(f"Found {len(job_configs)} job configs in DB.")

        for job_doc in job_configs:
            job_id = job_doc["job_id"]
            last_run_ts = job_doc.get("last_run_timestamp")

            # Find the most recent log entry for status
            latest_log = await asyncio.to_thread(
                logs_collection.find_one, {"job_id": job_id}, sort=[("timestamp", -1)]
            )

            status = "pending"
            error_message = None
            if latest_log:
                status = latest_log.get("status", "unknown")
                if status == "failure":
                    error_message = latest_log.get("stderr")
                elif latest_log.get("return_code", 0) != 0:
                    # Ensure non-zero exit codes are marked as failure
                    status = "failure"
                    error_message = latest_log.get("stderr", "Job exited non-zero.")

            # next_run is set to None as we don't calculate prediction here
            job_status = JobStatus(
                job_id=job_id,
                name=job_doc.get("name", ""),
                command=job_doc.get("command", ""),
                schedule=job_doc.get("schedule", ""),
                last_run=last_run_ts,
                next_run=None,  # Not calculated here
                telegram_chat_id=job_doc.get("telegram_chat_id"),
                status=status,
            )
            jobs.append(job_status)

        logger.debug(f"Returning {len(jobs)} job statuses.")
        return JobListResponse(jobs=jobs)
    except Exception as e:
        logger.error(f"Error listing jobs: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Error retrieving job list.")


@app.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(
    job_id: str,
    jobs_collection=Depends(get_jobs_collection),
    logs_collection=Depends(get_logs_collection),
):
    """Gets status of a specific job based on DB entry and latest log."""
    logger.debug(f"Request received for status of job: {job_id}")
    try:
        # Get job config from DB
        job_doc = await asyncio.to_thread(jobs_collection.find_one, {"job_id": job_id})
        if not job_doc:
            logger.warning(f"Job id '{job_id}' not found in database for status check.")
            raise HTTPException(
                status_code=404, detail="Job config not found in database"
            )

        last_run_ts = job_doc.get("last_run_timestamp")

        # Find the most recent log entry
        latest_log = await asyncio.to_thread(
            logs_collection.find_one, {"job_id": job_id}, sort=[("timestamp", -1)]
        )

        status = "pending"
        error_message = None
        if latest_log:
            status = latest_log.get("status", "unknown")
            if status == "failure":
                error_message = latest_log.get("stderr")
            elif latest_log.get("return_code", 0) != 0:
                status = "failure"
                error_message = latest_log.get("stderr", "Job exited non-zero.")

        logger.debug(f"Status for job {job_id}: {status}")
        return JobStatus(
            job_id=job_doc["job_id"],
            name=job_doc.get("name", ""),
            command=job_doc.get("command", ""),
            schedule=job_doc.get("schedule", ""),
            last_run=last_run_ts,
            next_run=None,  # Not calculated here
            telegram_chat_id=job_doc.get("telegram_chat_id"),
            status=status,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error getting status for job {job_id}: {e}\n{traceback.format_exc()}"
        )
        raise HTTPException(status_code=500, detail="Error retrieving job status.")


@app.delete("/jobs/{job_id}", response_model=DeleteJobResponse)
async def delete_job(
    job_id: str,
    scheduler: AsyncIOScheduler = Depends(get_scheduler),
    jobs_collection=Depends(get_jobs_collection),
    logs_collection=Depends(get_logs_collection),
):
    """Deletes a job from the database and unschedules it."""
    logger.info(f"Request received to delete job: {job_id}")

    # --- Unschedule Operation ---
    try:
        scheduler.remove_job(job_id)
        logger.info(f"Successfully unscheduled job: {job_id}")
    except Exception:
        logger.warning(
            f"Job {job_id} not found in scheduler for removal (might have already finished or been removed)."
        )
        # Continue to attempt DB deletion

    # --- Database Deletion ---
    try:
        delete_result = await asyncio.to_thread(
            jobs_collection.delete_one, {"job_id": job_id}
        )

        if delete_result.deleted_count == 0:
            logger.warning(
                f"Job config for {job_id} not found in database for deletion."
            )
            # If not found in scheduler AND not found in DB, return 404
            if scheduler.get_job(job_id):
                # If it's still in scheduler but not DB -> inconsistency, maybe 500?
                logger.error(
                    f"Inconsistency: Job {job_id} exists in scheduler but not DB."
                )
                raise HTTPException(
                    status_code=500,
                    detail="Job state inconsistent. Found in scheduler but not DB.",
                )
            else:
                # Not in scheduler, not in DB -> 404
                raise HTTPException(
                    status_code=404, detail="Job not found in scheduler or database."
                )

        logger.info(f"Successfully deleted job config from DB: {job_id}")
        return DeleteJobResponse(
            message="Job unscheduled and deleted successfully", deleted_job_id=job_id
        )
    except HTTPException:  # Re-raise 404 or 500 from above checks
        raise
    except Exception as e:
        logger.error(f"Error deleting job {job_id} from database: {e}")
        raise HTTPException(
            status_code=500, detail="Error deleting job config from database."
        )


@app.get("/")
async def root():
    scheduler = app_state.get("scheduler")
    status = "running" if scheduler and scheduler.running else "initializing_or_stopped"
    num_jobs = len(scheduler.get_jobs()) if scheduler else 0
    return {"status": status, "scheduled_jobs_count": num_jobs}


@app.on_event("startup")
async def startup_event():
    app.state.scheduler.start()
    app.state.scheduler_app = SchedulerApp()
    await app.state.scheduler_app.initialize()


@app.on_event("shutdown")
async def shutdown_event():
    if app.state.scheduler:
        app.state.scheduler.shutdown()
    if app.state.scheduler_app:
        await app.state.scheduler_app.stop()


# import uvicorn
# from .scheduler import app, scheduler_app


def main():
    """Run the FastAPI server."""
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=scheduler_app.settings.api_port if scheduler_app.settings else 18765,
        log_level="info",
    )


if __name__ == "__main__":
    main()
