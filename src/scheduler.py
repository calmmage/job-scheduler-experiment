from typing import List, Optional

from aiogram import Bot
from loguru import logger
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import SecretStr
from pydantic_settings import BaseSettings

from src.models import Job, JobCreate, JobUpdate


class SchedulerConfig(BaseSettings):
    """Scheduler configuration placeholder"""

    port: int = 18765  # Default port for the scheduler API

    mongodb_uri: SecretStr
    mongodb_db_name: str = "job_scheduler"

    # Telegram settings
    telegram_bot_token: SecretStr
    telegram_chat_id: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


class Scheduler:
    def __init__(self, config: SchedulerConfig):
        self.config = config
        self.bot = Bot(token=config.telegram_bot_token.get_secret_value())
        self.mongo_client = AsyncIOMotorClient(config.mongodb_uri.get_secret_value())
        self.db = self.mongo_client[config.mongodb_db_name]
        self.jobs_collection = self.db["jobs"]
        logger.info("Scheduler initialized with configuration")

    async def test_connection(self) -> bool:
        """Test MongoDB connection."""
        try:
            # The ismaster command is cheap and does not require auth
            await self.mongo_client.admin.command("ismaster")
            logger.info("MongoDB connection test successful")
            return True
        except Exception as e:
            logger.error(f"MongoDB connection test failed: {e}")
            return False

    async def add_job(self, job: JobCreate) -> Job:
        """Add a new job to the database."""
        job_dict = job.dict()
        job_dict["status"] = "pending"
        job_dict["retry_count"] = 0
        job_dict["last_run"] = None
        job_dict["next_run"] = None
        job_dict["error_message"] = None

        # Check if job with same key already exists
        existing = await self.jobs_collection.find_one({"job_key": job.job_key})
        if existing:
            raise ValueError(f"Job with key '{job.job_key}' already exists")

        result = await self.jobs_collection.insert_one(job_dict)
        if result.acknowledged:
            logger.info(f"Added job: {job.job_key}")
            return Job(**job_dict)
        else:
            raise Exception("Failed to add job")

    async def get_jobs(self) -> List[Job]:
        """Get all jobs from the database."""
        jobs = []
        cursor = self.jobs_collection.find({})
        async for document in cursor:
            jobs.append(Job(**document))
        return jobs

    async def get_job(self, job_key: str) -> Optional[Job]:
        """Get a specific job by key."""
        job = await self.jobs_collection.find_one({"job_key": job_key})
        if job:
            return Job(**job)
        return None

    async def update_job(self, job_key: str, job_update: JobUpdate) -> Optional[Job]:
        """Update an existing job."""
        # Remove None values
        update_data = {k: v for k, v in job_update.dict().items() if v is not None}
        if not update_data:
            return await self.get_job(job_key)

        result = await self.jobs_collection.update_one(
            {"job_key": job_key}, {"$set": update_data}
        )

        if result.matched_count:
            logger.info(f"Updated job: {job_key}")
            return await self.get_job(job_key)
        return None

    async def delete_job(self, job_key: str) -> bool:
        """Delete a job from the database."""
        result = await self.jobs_collection.delete_one({"job_key": job_key})
        if result.deleted_count:
            logger.info(f"Deleted job: {job_key}")
            return True
        return False
